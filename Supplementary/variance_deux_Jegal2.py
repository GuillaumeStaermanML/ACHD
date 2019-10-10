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
n = 100
Y = simul_simulate(n=n, m=m, T=1)
times = np.linspace(0,1,m)

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



l_sample = [20, 50, 100, 200, 500, 1000, 2000]


def boucle(l):
    score11 = np.zeros((200))
    score22 = np.zeros((200))
    score33 = np.zeros((200))
    score44 = np.zeros((200))
    np.random.seed(42)
    for k in range(100):
        Y = simul_simulate(n=l, m=m, T=1)
        FCHD1 = FCHD.FuncCHD(Y,times,)
        S1 = FCHD1.compute_depth(W)

        FCHD2 = FCHD.FuncCHD(Y,times, Subsampling=True, J=2)
        S2 = FCHD2.compute_depth(W)


        score11[k] = S1[0]
        score11[100+k] = S2[0]


        score22[k] = S1[1]
        score22[100+k] = S2[1]


        score33[k] = S1[2]
        score33[100+k] = S2[2]


        score44[k] = S1[3]
        score44[100+k] = S2[3]


    return score11, score22, score33, score44

if __name__ == '__main__': # excute on main process only
    #with Pool(4) as p:
    p = Pool(7)
    result0 = p.map(boucle, l_sample) 


AA = np.zeros((200,len(l_sample)))
BB = np.zeros((200,len(l_sample)))
CC = np.zeros((200,len(l_sample)))
DD = np.zeros((200,len(l_sample)))

for i in range(len(l_sample)):
	AA[:,i] = result0[i][0]

for i in range(len(l_sample)):
	BB[:,i] = result0[i][1]

for i in range(len(l_sample)):
	CC[:,i] = result0[i][2]  

for i in range(len(l_sample)):
	DD[:,i] = result0[i][3]     	

with open("variance_deux_0_Jegal2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(AA)
with open("variance_deux_1_Jegal2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(BB)
with open("variance_deux_2_Jegal2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(CC)
with open("variance_deux_3_Jegal2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(DD)