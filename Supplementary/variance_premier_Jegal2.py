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
m = 40
n = 100
X = simul_Brownien_geom(n=n,m=m,sigma=0.5,mu=2,T=1)
times = np.linspace(0,1,m)

Z = np.zeros((4,m))
Z[0] = X[99]
Z[1] = 6*times+np.cos(times*6*np.pi)
Z[2] = np.exp(3*times**2) + np.cos(times*2*np.pi)
Z[3] = X[9] 



l_sample = [20, 50, 100, 200, 500, 1000, 2000]


def boucle(l):
    score1 = np.zeros((200))
    score2 = np.zeros((200))
    score3 = np.zeros((200))
    score4 = np.zeros((200))
    np.random.seed(42)
    for k in range(100):
        Y = simul_Brownien_geom(n=l, m=m, T=1)
        FCHD1 = FCHD.FuncCHD(Y,times)
        S1 = FCHD1.compute_depth(Z)

        FCHD2 = FCHD.FuncCHD(Y,times, Subsampling=True, J=2)
        S2 = FCHD2.compute_depth(Z)



        score1[k] = S1[0]
        score1[100+k] = S2[0]


        score2[k] = S1[1]
        score2[100+k] = S2[1]
 

        score3[k] = S1[2]
        score3[100+k] = S2[2]


        score4[k] = S1[3]
        score4[100+k] = S2[3]


    return score1, score2, score3, score4

if __name__ == '__main__': # excute on main process only
    #with Pool(4) as p:
    p = Pool(7)
    result = p.map(boucle, l_sample) 


A = np.zeros((200,len(l_sample)))
B = np.zeros((200,len(l_sample)))
C = np.zeros((200,len(l_sample)))
D = np.zeros((200,len(l_sample)))


for i in range(len(l_sample)):
	A[:,i] = result[i][0]

for i in range(len(l_sample)):
	B[:,i] = result[i][1]

for i in range(len(l_sample)):
	C[:,i] = result[i][2]  

for i in range(len(l_sample)):
	D[:,i] = result[i][3]     	

with open("variance_premier_0_Jegal2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(A)
with open("variance_premier_1_Jegal2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(B)
with open("variance_premier_2_Jegal2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(C)
with open("variance_premier_3_Jegal2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(D)