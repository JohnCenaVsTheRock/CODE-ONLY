import sympy as sympy
import numpy as np
from numpy import linalg as LA
from sympy.physics.quantum.dagger import Dagger
from sympy import matrix2numpy
import matplotlib.pyplot as plt
# Python program to show time by process_time() 
from time import process_time

# produces the Matrix according to our hammiltonian
L = 270 # should be between 7-31 for N_1 + N_2 = 2*10 SO BASICALLY : (0.7 to 1.54)*Len_x*factor is recomended, best RESULTS with L =  (0.859)*Len_x*factor, DEFAULT 270
d_0 = 1.0 # DEFAULT 1.0
t_0 = 4.0 #  DEFAULT 4
delta_0 = 0.1 # DEFAULT 0.1
Hopping_lower = 1.0 # DEFAULT is 1.0 , speecify the inter chain hopping strangth here
Hopping_upper = 1.0 # DEFAULT is 1.0 , speecify the inter chain hopping strangth here

#set plot length in pixels
Len_y = 300 # DEFAULT 300
Len_x = 300 # is roughly half of the sum N_1+N_2 (provided factor is 1) DEFAULT 300
factor = int(1) # this factor determines the Matrix size independently of x (N_1/N_2 is equal to N_1*factor/N_2*factor) DEFAULT 1
Interval_y = [-8,8]# set intervals of the energy spectrum DEFAULT [-8,8]
# x interval is obsoltete because ammount of matrices is determined by pixel count Len_x
lambda_ = 0 # DEFAULT 0


# function for transition integral
def t(d,i,j,N_1,N_2):
    if(1<=i<= N_2 and 1<=j<=N_2): # intra chain hopping (lower chain)
        #print("both in lower chain")
        if(abs(i-j)==1): # nearest neighbour
            return [Hopping_lower,Hopping_lower] # returns [d_nummeric, d_symbolic]
        else: # same chain but far away
            return [0,0] # returns [d_nummeric, d_symbolic]
    elif(N_2<j<=N_1+N_2 and N_2<i<=N_1+N_2): # intra chain hopping (upper chain)
        #print("both in upper chain")
        if(abs(i-j)==1): # nearest neighbour
            return [Hopping_upper,Hopping_upper] # returns [d_nummeric, d_symbolic]
        else: # same chain but far away
            return [0,0] # returns [d_nummeric, d_symbolic]
    else:
        return np.array([sympy.N((t_0*sympy.exp(-(d-d_0)/delta_0))), t_0*sympy.exp(-(d-d_0)/delta_0)]) # returns [d_nummeric, d_symbolic]

# function for euclidean distance
def distance(i,j,lambda_, N_1, N_2): # [lambda_] shifts the upper chain in x dir by lambda_*L
    a_1 = L/(N_1-1)
    a_2 = L/(N_2-1)
    v_i = [0,0] #  coordinates in 2d space of lattice i
    v_j = [0,0] #  coordinates in 2d space of lattice j
    if(1<=i<=N_2): # i is in lower chain
        v_i[0] = a_2*(i-1) # x coordinate of lattice i
        v_i[1] = 0 # y coordinate of lattice i
    if(N_2<i<=N_1+N_2): # i is in upper chain
        v_i[0] = a_1*(i-N_2-1) + lambda_*a_1 # x coordinate of lattice i
        v_i[1] = d_0 # y coordinate of lattice i
    if(1<=j<=N_2): # j is in lower chain
        v_j[0] = a_2*(j-1) # x coordinate of lattice j
        v_j[1] = 0 # y coordinate of lattice j
    if(N_2<j<=N_1+N_2): # j is in upper chain
        v_j[0] = a_1*(j-N_2-1) + lambda_*a_1 # x coordinate of lattice j
        v_j[1] = d_0 # y coordinate of lattice j
    return (abs((v_i[0]-v_j[0])**2+(v_i[1]-v_j[1])**2))**0.5 # distance between these vectors

# cycle trought Matrix and populate it
def calculateMatrices(lambda_, N_1, N_2, symbolic_calculating):
    t1_start = process_time()  # cpu time start
    M_nummeric = np.zeros(((N_2+N_1), (N_2+N_1)))
    M_symbolic = sympy.zeros((N_2+N_1), (N_2+N_1))
    for i in range(1,N_1+N_2 + 1): # rows
        for j in range(i+1,N_1+N_2 + 1): # cols (but beginning at i such that we get upper traing matrix)
                #print(i)
                #print(j)
                d = distance(i,j,lambda_,N_1,N_2)
                M_nummeric[i-1,j-1] = t(d,i,j,N_1,N_2)[0]
                M_symbolic[i-1,j-1] = t(d,i,j,N_1,N_2)[1]
    M_nummeric = (M_nummeric + np.transpose(np.conjugate(M_nummeric))) # holds the numpy Hamiltonian (better for approximations)
    M_symbolic = (M_symbolic + Dagger(M_symbolic)) # holds the sympy Hamiltonian (better for diagonalization)
    # FOR TESTING
    global M
    M = M_nummeric
    if(symbolic_calculating):
        P, D = M_symbolic.diagonalize() # diagonalizing
        Diagonal = matrix2numpy(D, dtype=float)        
    else:
        D = LA.eigvals(M_nummeric)
        Diagonal = np.diag(D)
    # just a placeholder in this program
    PsiOfX = "placeholder"
    t1_stop = process_time() # cpu time stop
    print("time elpased for calculateMatrices for lambda = ", lambda_, ", N_1 = ", N_1, ", N_2 = ", N_2, " : ", -(t1_start-t1_stop))
    return [PsiOfX, Diagonal]




#function for populating image here
def populate_col_to_image(img, img2, img_chern, lambda_, N_1, N_2, col, symbolic_calculating): # calculates the eigenvalues and appends them to the img vector as a new column
    calculations = calculateMatrices(lambda_, N_1, N_2, symbolic_calculating) #enter the shift and the numbers N_1, N_2
    diag_ = calculations[1].diagonal() # holds the diagonal i.e the eigenvectors
    #print("Diagonal matrix, D: ", diag_)
    diag_ = diag_[(diag_ >= Interval_y[0]) & (diag_ <= Interval_y[1])] # remove all values taht are outside of specifyed range Inetrval_y[.,.]
    #print("D, valueas in range: ", diag_)
    # now we want so scale and translate the vector so that the Interval_y[0] maps to Len_y and Interval_y[1] to 0
    diag_ = (diag_+Interval_y[1])*(Len_y/(Interval_y[1]-Interval_y[0]))
    #print("D, scaled: ", diag_)
    # rounding and converting to integers
    diag_ = np.round(diag_).astype(int)
    #print("D, rounded: ",diag_)
    # create new matrix first, to avoid array problems later
    img_chern_2 = np.zeros((Len_y+1,Len_x,4)).astype(np.float32)
    # now we populate the img matrix accordingly
    for i in range(0,diag_.size):
        img[len(img) - 1 - diag_[i]][col-1][0] = 0.0 # matrix red tone
        img[len(img) - 1 - diag_[i]][col-1][1] = 0.0 # matrix green tone
        img[len(img) - 1 - diag_[i]][col-1][2] = 1 # matrix blue tone
        img[len(img) - 1 - diag_[i]][col-1][3] += 1/max(np.bincount(diag_))   # matrix opacity (will find the eigenvalue in diag_ that occurs most often and add 1/ammount_of_occurances to opacity each time the pixel is hit)
        # populate additional multidim matrix img_chern that stores r (band ammount underneath) and N_2, and alpha
        # ++++ TO BE DONE ++++ the WHOLE matrix has to be populated, not just energy band area
        img2[len(img) - 1 - diag_[i]][col-1][0] = 0.0 # matrix red tone
        img2[len(img) - 1 - diag_[i]][col-1][1] = 0.0 # matrix green tone
        img2[len(img) - 1 - diag_[i]][col-1][2] = 1.0 # matrix blue tone
        img2[len(img) - 1 - diag_[i]][col-1][3] = 1.0   # matrix opacity (will find the eigenvalue in diag_ that occurs most often and add 1/ammount_of_occurances to opacity each time the pixel is hit)
        #
        img_chern_2[len(img) - 1 - diag_[i]][col-1][1] += 1 # r value
    # reiterate the WHOLE matrix img_chern, in order to calculate r values correctly
    for row_1 in range(0, len(img_chern_2)): # only rows are needed, since iteration trought cols already takes place
        summ_r = 0 # summating here
        for i in range (0, row_1):
            #print("\n")
            summ_r += img_chern_2[len(img) - 1- i][col-1][1]
            #print(i)
            #print(img_chern_2[i][col-1][1])
            img_chern[len(img) - 1- i][col-1][1] = summ_r
            #print(img_chern[i][col-1][1])
        img_chern[len(img) - 1 - row_1][col-1][2] = float(N_2) # N_2 value
        img_chern[len(img) - 1 - row_1][col-1][3] = float(N_1)/float(N_2) # alpha value

def calculate_Spectrum(img_chern, Len_y, Len_x, Interval_y, factor, lambda_, constant_precision, symbolic_calculating):
    # create image array
    print("calculating Spectrum now...")
    img = np.zeros((Len_y+1,Len_x,4)).astype(np.float32) # Len_y+1 TO BE DONE (dont know)
    img2 = np.zeros((Len_y+1,Len_x,4)).astype(np.float32) # Len_y+1 TO BE DONE (dont know)
    # store additional information for eventual chern number calculations
    #img_chern = np.zeros((Len_y+1,Len_x,4)).astype(np.float32)
    # iterate over the columns to populate full image
    if(constant_precision):# constant precision ensures equal ammounts of eigenvalues for all columns
        for i in range(2,Len_x+1): # cicles Len_x times (starts at 2 because N_1 = 0 or 1 makes no sense) [a_1 is infty if N_1 = 1] 
            N_1 = int(np.trunc((((2*Len_x*(1-1/(1+i/Len_x))))*factor))) #  np.trunc to round down to avoid overflows
            N_2 = int(np.trunc((((2*Len_x)/(1+i/Len_x))*factor)))
            # exeptions handeled here
            if(N_1 == 0):
                N_1 = 2
            if(N_2 == 0):
                N_2 = 2
            if(N_1 == 1):
                N_1 = 2
            if(N_2 == 1):
                N_2 = 2
            print("high precision with N_1 = ", N_1, " N_2 = ", N_2)
            populate_col_to_image(img, img2, img_chern, lambda_, N_1, N_2, i, symbolic_calculating) # populates the column i of the img matrix
    else:
        for i in range(1,Len_x+1): # cicles Len_x times (starts at 1 because N_1 = 0 makes no sense)
            populate_col_to_image(img, img2, img_chern, lambda_, i*factor, (Len_x)*factor, i, symbolic_calculating) # populates the column i of the img matrix
    # plot the array img
    plt.imshow(img)
    # plotting starts here
    #save the image
    #plt.savefig("lower_hopping_param={}_upper_hopping_param={}.svg".format(Hopping_lower, Hopping_upper), bbox_inches='tight') # save a svg to folder
    if(Grid_size == 2):
        plt.xlabel('$N_1/N_2$')  
        plt.ylabel('$E$')  
        y_tiks_interval = np.round(np.linspace(0, Len_y, 9), 2)
        plt.yticks(y_tiks_interval, labels=np.round(np.linspace(Interval_y[1],Interval_y[0], 9),1))
        x_tiks_interval = np.round(np.linspace(0, Len_x, 10), 1)
        plt.xticks(x_tiks_interval, labels=np.round(np.arange(0, 1, 0.1), 2))
        plt.savefig("CO_Spectrum.svg") # save a svg to folder
    else:
        # Disable tiks for now
        plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
        plt.savefig("different_energy_scale_lower_hopping_param={}_upper_hopping_param={}.svg".format(Hopping_lower, Hopping_upper), bbox_inches='tight') # save a svg to folder
    plt.show()
    #
    #
    #
    #
    # plot the array img2
    plt.imshow(img2)
    # plotting starts here
    if(Grid_size == 2):
        plt.xlabel('$N_1/N_2$')  
        plt.ylabel('$E$')  
        y_tiks_interval = np.round(np.linspace(0, Len_y, 9), 2)
        plt.yticks(y_tiks_interval, labels=np.round(np.linspace(Interval_y[1],Interval_y[0], 9),1))
        x_tiks_interval = np.round(np.linspace(0, Len_x, 10), 1)
        plt.xticks(x_tiks_interval, labels=np.round(np.arange(0, 1, 0.1), 2))
        plt.savefig("Spectrum.svg") # save a svg to folder
    else:
        # Disable tiks for now
        plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
        plt.savefig("Spectrum_different_energy_scale_lower_hopping_param={}_upper_hopping_param={}.svg".format(Hopping_lower, Hopping_upper), bbox_inches='tight') # save a svg to folder
    plt.show()
    #
    #
    #
    # plotting Chern Numbers here
    print("calculating Chern Numbers now...")
    # create new matrix to avoid array problems later
    img_chern_3 = img_chern.copy()
    for row in range(0, len(img_chern_3)-1):
        for col in range(0, len(img_chern_3[row])-1):
            r_1 =  float(img_chern_3[row][col][1])
            r_2 =  float(img_chern_3[row][col + 1][1])
            alpha_1 =  float(img_chern_3[row][col][3])
            alpha_2 =  float(img_chern_3[row][col + 1][3])
            N_2_1 =  float(img_chern_3[row][col][2])
            N_2_2 =  float(img_chern_3[row][col + 1][2])
            if(N_2_1 == 0 or N_2_2 == 0):
                Chern_Number_1 = float("inf")
            elif(alpha_2 != alpha_1): # can happen at borders
                Chern_Number_1 = ((r_2/N_2_2) - (r_1/N_2_1))/(alpha_2-alpha_1)
                Chern_Number_2 = (r_1-alpha_1*N_2_1*Chern_Number_1)/N_2_1 # C_2 is easy to calculate by manipulating the diophane equation after you figured out C_1
            # Chern Numbers
            if(Chern_Number_1 == float("inf") or Chern_Number_2 == float("inf")):
                img_chern[row][col][3] = 0.0 # opacity
            else:
                img_chern[row][col][3] = 1.0 # opacity
                #img_chern[row][col][0] = float(0.5 + 0.2*np.round(Chern_Number_1)) # red tone representing chern_number_1 = 3
                img_chern[row][col][0] = 0.5+0.5*(np.arctan(np.round((Chern_Number_1)))*(2/np.pi)) # red tone representing chern_number_1
                img_chern[row][col][1] = 0 # np.round(Chern_Number_2) # green tone representing chern_number_1 = 3
                #img_chern[row][col][2] = float(0.5 + 0.2*np.round(Chern_Number_2)) # blue tone representing chern number 2
                img_chern[row][col][2] =0.5+0.5*(np.arctan(np.round((Chern_Number_2)))*(2/np.pi)) # blue tone representing chern_number_2
    plt.imshow(img_chern) 
    #save the image
    #plt.savefig("lower_hopping_param={}_upper_hopping_param={}.svg".format(Hopping_lower, Hopping_upper), bbox_inches='tight') # save a svg to folder
    if(Grid_size == 2):
        y_tiks_interval = np.round(np.linspace(0, Len_y, 9), 2)
        plt.yticks(y_tiks_interval, labels=np.round(np.linspace(Interval_y[1],Interval_y[0], 9),1))
        x_tiks_interval = np.round(np.linspace(0, Len_x, 10), 1)
        plt.xticks(x_tiks_interval, labels=np.round(np.arange(0, 1, 0.1), 2))
        plt.xlabel('$N_1/N_2$')  
        plt.ylabel('$E$') 
        plt.savefig("CHERN_NUMBERS_different_energy_scale_lower_hopping_param=1.0_upper_hopping_param=1.0.svg") # save a svg to folder
    else:
        # Disable tiks for now
        plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
        plt.savefig("CHERN_NUMBERS_different_energy_scale_lower_hopping_param={}_upper_hopping_param={}.svg".format(Hopping_lower, Hopping_upper), bbox_inches='tight') # save a svg to folder
    plt.show()
    #
    #
    #
    # plotting the scaling-Legend for the chern numbers here
    img_Legend = np.zeros((100,100,4)).astype(np.float32)
    smoothness = 20 # determines visible borders in discretisation
    for row in range(0, len(img_Legend)):
        for col in range(len(img_Legend[0])):
            img_Legend[row][len(img_Legend) - 1 - col][0] = 0.5+0.5*(np.sign(row-50)*np.arctan(np.round(((row-50)/smoothness)**2)**0.5)*(2/np.pi)) # red
            img_Legend[row][len(img_Legend) - 1 - col][2] = 0.5+0.5*(np.sign(col-50)*np.arctan(np.round(((col-50)/smoothness)**2)**0.5)*(2/np.pi)) # blue
            img_Legend[row][len(img_Legend) - 1 - col][3] = 1.0 # opacity
    plt.imshow(img_Legend)
    x_tiks_interval = [70,50,30]
    plt.xticks(x_tiks_interval, labels=[-1, 0, 1])
    y_tiks_interval = [30,50,70]
    plt.yticks(y_tiks_interval, labels=[-1, 0, 1])
    plt.xlabel('$C_2$')  
    plt.ylabel('$C_1$')  
    plt.savefig("SCALING.svg", bbox_inches='tight') # save a svg to folder
    plt.show()





# main program here:
# if you only want one detailed image set Grid_size to 2
Grid_size = 2 #default 4 (3x3 Grid)
for i in range(1,Grid_size):
    for j in range(1,Grid_size):
        # print("i:")
        # print(i)
        # print("j:")
        # print(j)
        Hopping_lower = i
        Hopping_upper = j
        img_chern = np.zeros((Len_y+1,Len_x,4)).astype(np.float32)
        # run main plotting function here
        calculate_Spectrum(img_chern, Len_y, Len_x, Interval_y, factor, lambda_, True, False) # constant_precision determines wether you go from (N_1 = 1, N_2 = L), to (N_1 = L, N_2 = L), or wether you go from (2Li/(1+i=1), 2L/(1+i=1)) to (N_1 = L, N_2 = L)
Chern_Number_1 = -1
print(0.5+(np.sign(Chern_Number_1)*np.arctan(np.floor((Chern_Number_1)))*(2/np.pi)))