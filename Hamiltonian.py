import sympy as sympy
import numpy as np
from sympy.physics.quantum.dagger import Dagger
from sympy import matrix2numpy
import matplotlib.pyplot as plt

# produces the Matrix according to our hammiltonian
L = 30
d_0 = 1.0
t_0 = 4.0
delta_0 = 0.1
N_1 = 11
N_2 = 10
a_1 = L/N_1
a_2 = L/N_2

# FOR TESTING
M = np.zeros(((N_2+N_1), (N_2+N_1)))

# function for transition integral
def t(d,i,j):
    if(1<=i<= N_2 and 1<=j<=N_2): # intra chain hopping (lower chain)
        #print("both in lower chain")
        if(abs(i-j)==1): # nearest neighbour
            return [1,1] # returns [d_nummeric, d_symbolic]
        else: # same chain but far away
            return [0,0] # returns [d_nummeric, d_symbolic]
    elif(N_2<j<=N_1+N_2 and N_2<i<=N_1+N_2): # intra chain hopping (upper chain)
        #print("both in upper chain")
        if(abs(i-j)==1): # nearest neighbour
            return [1,1] # returns [d_nummeric, d_symbolic]
        else: # same chain but far away
            return [0,0] # returns [d_nummeric, d_symbolic]
    else:
        return np.array([sympy.N((t_0*sympy.exp(-(d-d_0)/delta_0))), t_0*sympy.exp(-(d-d_0)/delta_0)]) # returns [d_nummeric, d_symbolic]

# function for euclidean distance
def distance(i,j,lambda_): # [lambda_] shifts the upper chain in x dir by lambda_*L
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

# cycle trought Matrix and fill it up
def calculateMatrices(lambda_):
    M_nummeric = np.zeros(((N_2+N_1), (N_2+N_1)))
    M_symbolic = sympy.zeros((N_2+N_1), (N_2+N_1))
    for i in range(1,N_1+N_2 + 1): # rows
        for j in range(i+1,N_1+N_2 + 1): # cols (but beginning at i such that we get upper traing matrix)
                #print(i)
                #print(j)
                d = distance(i,j,lambda_)
                M_nummeric[i-1,j-1] = t(d,i,j)[0]
                M_symbolic[i-1,j-1] = t(d,i,j)[1]
    M_nummeric = (M_nummeric + np.transpose(np.conjugate(M_nummeric))) # holds the numpy Hamiltonian (better for approximations)
    M_symbolic = (M_symbolic + Dagger(M_symbolic)) # holds the sympy Hamiltonian (better for diagonalization)
    # FOR TESTING
    global M
    M = M_nummeric
    P, D = M_symbolic.diagonalize() # diagonalizing
    Eigenvectors = matrix2numpy(P, dtype=float) 
    Diagonal = matrix2numpy(D, dtype=float) 
    # taking the sqare of P entry wise to get psi(R_i)
    PsiOfX = abs(Eigenvectors)**2
    return [PsiOfX, Diagonal]



lambda_ = 0
# plotting the first eigenvector (some fix energy) chain 2
calculations = calculateMatrices(lambda_)
# look for the best eigenvalue to konwo wich eigenfuction to choose
Eigenvalue = np.argmin(calculations[1].diagonal()) # the eigenvalue that should be considered
xpoints_2 = np.linspace(0, 1, num=(N_2))
ypoints_2 = np.transpose(calculations[0])[Eigenvalue][0:N_2]
plt.subplot(3, 1, 1)
plt.plot(xpoints_2, ypoints_2, label="chain 2 (lower)")
# plotting the best eigenvector (some fix energy) chain 1
xpoints_1 = np.linspace(0, 1, num=(N_1))
xpoints_1 = xpoints_1+(a_1*lambda_)/L # mitigating the shift
ypoints_1 = np.transpose(calculations[0])[Eigenvalue][N_2:N_2+N_1]
plt.subplot(3, 1, 1)
plt.xticks([])
plt.yticks([])
plt.plot(xpoints_1, ypoints_1, label="chain 1 (upper)")
plt.legend(loc="upper right")
plt.title(f"$\lambda = ${lambda_}", x=-0.1, y=0.11) 


lambda_ = 1/4
# plotting the best eigenvector (some fix energy) chain 2
calculations = calculateMatrices(lambda_)
# look for the lowest eigenvalue to konwo wich eigenfuction to choose
Eigenvalue =  np.argmin(calculations[1].diagonal()) # the eigenvalue that should be considered
xpoints_2 = np.linspace(0, 1, num=(N_2))
ypoints_2 = np.transpose(calculations[0])[Eigenvalue][0:N_2]
plt.subplot(3, 1, 2)
plt.plot(xpoints_2, ypoints_2)
# plotting the best eigenvector (some fix energy) chain 1
xpoints_1 = np.linspace(0, 1, num=(N_1))
xpoints_1 = xpoints_1+(a_1*lambda_)/L # mitigating the shift
ypoints_1 = np.transpose(calculations[0])[Eigenvalue][N_2:N_2+N_1]
plt.subplot(3, 1, 2)
plt.xticks([])
plt.yticks([])
plt.plot(xpoints_1, ypoints_1)
plt.title(f"$\lambda = ${lambda_}", x=-0.1, y=0.11) 

lambda_ = 1/2
# plotting the best eigenvector (some fix energy) chain 2
calculations = calculateMatrices(lambda_)
# look for the lowest eigenvalue to konwo wich eigenfuction to choose
Eigenvalue =  np.argmin(calculations[1].diagonal()) # the eigenvalue that should be considered
xpoints_2 = np.linspace(0, 1, num=(N_2))
ypoints_2 = np.transpose(calculations[0])[Eigenvalue][0:N_2]
plt.subplot(3, 1, 3)
plt.plot(xpoints_2, ypoints_2)
# plotting the best eigenvector (some fix energy) chain 1
xpoints_1 = np.linspace(0, 1, num=(N_1))
xpoints_1 = xpoints_1+(a_1*lambda_)/L # mitigating the shift
ypoints_1 = np.transpose(calculations[0])[Eigenvalue][N_2:N_2+N_1]
plt.subplot(3, 1, 3)
plt.xticks([])
plt.yticks([])
plt.xlabel("$ x/(L+a_1\lambda)$")
plt.plot(xpoints_1, ypoints_1)
plt.title(f"$\lambda = ${lambda_}", x=-0.1, y=0.11) 
plt.savefig("twoChanisPlot.svg")
plt.show()
