import FCHD
import csv
from multiprocessing import Pool
import numpy as np

def simul_Brownien_geom(n = 100,m = 1000,sigma = 0.5,mu = 2, T = 1):
    tps = np.linspace(0,T,m) # Discrétisation du temps
    B = np.zeros((n,m))
    B[:,0] = 0 # départ du MB à 0
    S = np.zeros((n,m))
    S[:,0] = 1 # départ du BG à 1
    S0=1 # Constante d'initialisation du brownien géométrique
    for i in range(1,np.size(tps)):
        B[:,i] = B[:, i-1] + sigma*np.random.normal(0,np.sqrt(tps[2]-tps[1]),n)+ mu*(tps[2]-tps[1]) 
        S[:,i] = S0*np.exp(B[:,i]) # Translation par rapport à la moyenne sur la formule originale t(mu- sigma^2/2)

    return S;

np.random.seed(42)
n = 200
m = 40
X = simul_Brownien_geom(n=n,m=m,sigma=0.5,mu=2,T=1)

times = np.linspace(0,1,m)

Z = np.zeros((4,m))
Z[0] = X[99]
Z[1] = 6*times+np.cos(times*6*np.pi)
Z[2] = np.exp(3*times**2) + np.cos(times*2*np.pi)
Z[3] = X[9] 

a = [0, 3]

liste_K = [1, 2, 3, 4, 5, 10, 20, 35, 50, 75]
nl = len(liste_K)

def Kimpact(liste):
    np.random.seed(42)

    score = np.zeros((6,100))


    for k in range(100):
        FCHD1 = FCHD.FuncCHD(X, times,  Subsampling=True, K= liste * n)    
        S_exact1 = FCHD1.compute_depth(Z[a])
        FCHD2 = FCHD.FuncCHD(X, times,  Subsampling=True, K= liste * n, J=3)    
        S_exact2 = FCHD2.compute_depth(Z[a])
        FCHD3 = FCHD.FuncCHD(X, times,  Subsampling=True, K= liste * n, J=4)    
        S_exact3 = FCHD3.compute_depth(Z[a])

        score[0,k] = S_exact1[0]
        score[1,k] = S_exact2[0]
        score[2,k] = S_exact3[0]
        score[3,k] = S_exact1[1]
        score[4,k] = S_exact2[1]
        score[5,k] = S_exact3[1]

    return score

if __name__ == '__main__': # excute on main process only
    #with Pool(4) as p:
    p = Pool(10)
    result2 = p.map(Kimpact, liste_K) 

J_2 = np.zeros((101, 2 * nl))
J_3 = np.zeros((101, 2 * nl))
J_4 = np.zeros((101, 2 * nl))

for i in range(len(liste_K)):
    J_2[1:,i] = result2[i][0]
    J_2[1:,nl+i] = result2[i][3]
    J_3[1:,i] = result2[i][1]
    J_3[1:,nl+i] = result2[i][4]
    J_4[1:,i] = result2[i][2]
    J_4[1:,nl+i] = result2[i][5]





FCHD2 = FCHD.FuncCHD(X, times, )    
S_exact = FCHD2.compute_depth(Z[a])
J_2[0, :nl] = S_exact[0]
J_2[0, nl:] = S_exact[1]

FCHD2 = FCHD.FuncCHD(X, times,  J=3)    
S_exact = FCHD2.compute_depth(Z[a])
J_3[0, :nl] = S_exact[0]
J_3[0, nl:] = S_exact[1]

FCHD2 = FCHD.FuncCHD(X, times, J=4)    
S_exact = FCHD2.compute_depth(Z[a])
J_4[0, :nl] = S_exact[0]
J_4[0, nl:] = S_exact[1]

B = np.concatenate((J_2,J_3,J_4), axis =1)


with open("impact_K_premier.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(B)
