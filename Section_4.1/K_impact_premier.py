import FCHD
import csv
from multiprocessing import Pool
import numpy as np


def simul_simulate(n = 100, m = 200, T = 1):
    tps = np.linspace(0,T,m) # Discrétisation du temps
    B = np.zeros((n,m))
    a1 = np.zeros((n))
    a2 = np.zeros((n))
    for i in range(0,n):
        a1[i] = 0.05 * np.random.random()
        a2[i] = 0.05 * np.random.random()
        for j in range(0,m):
                B[i,j] = a1[i] * np.cos(tps[j] *
                 2 * np.pi) + a2[i] * np.sin(tps[j] * 2 * np.pi)
    return B;

np.random.seed(42)
m = 40
n = 200
Y = simul_simulate(n=n, m=m, T=1)
times = np.linspace(0,1,m)
a = [0, 3]
W = np.zeros((4,m))
for j in range(0,m):
    W[0,j] = 0.025 * np.cos(times[j] *
     2 * np.pi) + 0.025 * np.sin(times[j] * 2 * np.pi)
    W[1,j] = 0.025 * np.cos(times[j] *
     2 * np.pi) + 0.025 * np.sin(times[j] * 2 * np.pi)+np.random.normal(0,0.005)
    W[2,j] = 0.055 * np.cos(times[j] *
     2 * np.pi) + 0.055 * np.sin(times[j] * 2 * np.pi)
    W[3,j] = 0.055 * np.cos((times[j]+0.5)  *
     2 * np.pi+0.5) + 0.055 * np.sin((times[j]+0.5) * 2 * np.pi)


liste_K = [1, 2, 3, 4, 5, 10, 20, 35, 50, 75]
nl = len(liste_K)

def Kimpact(liste):
    np.random.seed(42)

    score = np.zeros((6,100))


    for k in range(100):
        FCHD1 = FCHD.FuncCHD(Y, times, Subsampling=True, K= liste * n)    
        S_exact1 = FCHD1.compute_depth(W[a])
        FCHD2 = FCHD.FuncCHD(Y, times,  Subsampling=True, K= liste * n, J=3)    
        S_exact2 = FCHD2.compute_depth(W[a])
        FCHD3 = FCHD.FuncCHD(Y, times,  Subsampling=True, K= liste * n, J=4)    
        S_exact3 = FCHD3.compute_depth(W[a])

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
    result = p.map(Kimpact, liste_K) 

J_2 = np.zeros((101, 2 * nl))
J_3 = np.zeros((101, 2 * nl))
J_4 = np.zeros((101, 2 * nl))

for i in range(len(liste_K)):
    J_2[1:,i] = result[i][0]
    J_2[1:,nl+i] = result[i][3]
    J_3[1:,i] = result[i][1]
    J_3[1:,nl+i] = result[i][4]
    J_4[1:,i] = result[i][2]
    J_4[1:,nl+i] = result[i][5]



FCHD2 = FCHD.FuncCHD(Y, times)    
S_exact = FCHD2.compute_depth(W[a])
J_2[0, :nl] = S_exact[0]
J_2[0, nl:] = S_exact[1]

FCHD2 = FCHD.FuncCHD(Y, times,  J=3)    
S_exact = FCHD2.compute_depth(W[a])
J_3[0, :nl] = S_exact[0]
J_3[0, nl:] = S_exact[1]

FCHD2 = FCHD.FuncCHD(Y, times,  J=4)    
S_exact = FCHD2.compute_depth(W[a])
J_4[0, :nl] = S_exact[0]
J_4[0, nl:] = S_exact[1]

A = np.concatenate((J_2,J_3,J_4), axis =1)

with open("impact_K_deux.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(A)
