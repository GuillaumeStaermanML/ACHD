import numpy as np
from math import *
import pandas as pd
import csv
import FCHD
import FIF

def functional_Stahel_Donoho(X_train, X_test):
    m = X_train.shape[1]
    n1 = X_train.shape[0]
    n2 = X_test.shape[0]
    S = np.zeros((n2,m))
    for t in range(m):
        for i in range(n2):
            S[i,t] = np.abs(X_test[i,t] - np.median(X_train[:,t]))
        if (np.median(np.abs(X_train[:,t]-np.median(X_train[:,t]))) > 0):
            S[:,t] = 1 / (1 + S[:,t] / np.median(np.abs(X_train[:,t]-np.median(X_train[:,t]))))
    
    return  np.mean(S, axis=1) 

def functional_Tukey(X_train, X_test):
    m = X_train.shape[1]
    n1 = X_train.shape[0]
    n2 = X_test.shape[0]
    S = np.zeros((n2,m))
    Z = np.zeros(n1)
    for t in range(m):
        for i in range(n2):
            Z = X_train[:,t] - X_test[i, t]
            S[i, t] = np.sum(1* (Z >0))
        S[:, t] = S[:, t]  / n1
        S[:, t]  = np.minimum(S[:, t] , 1 - S[:, t]  )
    return np.mean(S, axis=1)


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


n = 100
m = 40

X = pd.read_csv('Dataset_deux.csv', header=None)
X = X.as_matrix()
X = X[:100]
np.random.seed(42)
times = np.linspace(0,1,m)
Z = np.zeros((30, X.shape[1]))
for i in range(30):
    Z[i] = X[1] + X[1] + np.random.uniform(0.035,0.1)

X1 = np.concatenate((X,Z[:5]))
X2 = np.concatenate((X,Z[:10]))
X3 = np.concatenate((X,Z[:15]))
X4 = np.concatenate((X,Z[:25]))
X5 = np.concatenate((X,Z[:30]))


un_FCHD0 = FCHD.FuncCHD(X,times)
un_Score0 = un_FCHD0.compute_depth(X)
un_FCHD1 = FCHD.FuncCHD(X1,times)
un_Score1 = un_FCHD1.compute_depth(X)
un_FCHD2 = FCHD.FuncCHD(X2,times)
un_Score2 = un_FCHD2.compute_depth(X)
un_FCHD3 = FCHD.FuncCHD(X3,times)
un_Score3 = un_FCHD3.compute_depth(X)
un_FCHD4 = FCHD.FuncCHD(X4,times)
un_Score4 = un_FCHD4.compute_depth(X)
un_FCHD5 = FCHD.FuncCHD(X5,times)
un_Score5 = un_FCHD5.compute_depth(X)

un_FSDO0 = functional_Stahel_Donoho(X, X)
un_FSDO1 = functional_Stahel_Donoho(X1, X)
un_FSDO2 = functional_Stahel_Donoho(X2, X)
un_FSDO3 = functional_Stahel_Donoho(X3, X)
un_FSDO4 = functional_Stahel_Donoho(X4, X)
un_FSDO5 = functional_Stahel_Donoho(X5, X)

un_FT0 = functional_Tukey(X, X)
un_FT1 = functional_Tukey(X1, X)
un_FT2 = functional_Tukey(X2, X)
un_FT3 = functional_Tukey(X3, X)
un_FT4 = functional_Tukey(X4, X)
un_FT5 = functional_Tukey(X5, X)

FIF0 = FIF.FIForest(X, time=times, innerproduct='auto', D='cosinus')
un_FIF0 = 1 - FIF0.compute_paths(X)
FIF1 = FIF.FIForest(X1, time=times, innerproduct='auto', D='cosinus')
un_FIF1 = 1 - FIF1.compute_paths(X)
FIF2 = FIF.FIForest(X2, time=times, innerproduct='auto', D='cosinus')
un_FIF2 = 1 - FIF2.compute_paths(X)
FIF3 = FIF.FIForest(X3, time=times, innerproduct='auto', D='cosinus')
un_FIF3 = 1 - FIF3.compute_paths(X)
FIF4 = FIF.FIForest(X4, time=times, innerproduct='auto', D='cosinus')
un_FIF4 = 1 - FIF4.compute_paths(X)
FIF5 = FIF.FIForest(X5, time=times, innerproduct='auto', D='cosinus')
un_FIF5 = 1 - FIF5.compute_paths(X)

for i in range(9):
    X = simul_simulate(n, m)

    X1 = np.concatenate((X,Z[:5]))
    X2 = np.concatenate((X,Z[:10]))
    X3 = np.concatenate((X,Z[:15]))
    X4 = np.concatenate((X,Z[:25]))
    X5 = np.concatenate((X,Z[:30]))


    un_FCHD0 = FCHD.FuncCHD(X,times)
    un_Score0 += un_FCHD0.compute_depth(X)
    un_FCHD1 = FCHD.FuncCHD(X1,times)
    un_Score1 += un_FCHD1.compute_depth(X)
    un_FCHD2 = FCHD.FuncCHD(X2,times)
    un_Score2 += un_FCHD2.compute_depth(X)
    un_FCHD3 = FCHD.FuncCHD(X3,times)
    un_Score3 += un_FCHD3.compute_depth(X)
    un_FCHD4 = FCHD.FuncCHD(X4,times)
    un_Score4 += un_FCHD4.compute_depth(X)
    un_FCHD5 = FCHD.FuncCHD(X5,times)
    un_Score5 += un_FCHD5.compute_depth(X)

    un_FSDO0 += functional_Stahel_Donoho(X, X)
    un_FSDO1 += functional_Stahel_Donoho(X1, X)
    un_FSDO2 += functional_Stahel_Donoho(X2, X)
    un_FSDO3 += functional_Stahel_Donoho(X3, X)
    un_FSDO4 += functional_Stahel_Donoho(X4, X)
    un_FSDO5 += functional_Stahel_Donoho(X5, X)

    un_FT0 += functional_Tukey(X, X)
    un_FT1 += functional_Tukey(X1, X)
    un_FT2 += functional_Tukey(X2, X)
    un_FT3 += functional_Tukey(X3, X)
    un_FT4 += functional_Tukey(X4, X)
    un_FT5 += functional_Tukey(X5, X)

    FIF0 = FIF.FIForest(X, time=times, innerproduct='auto', D='cosinus')
    un_FIF0 += 1 - FIF0.compute_paths(X)
    FIF1 = FIF.FIForest(X1, time=times, innerproduct='auto', D='cosinus')
    un_FIF1 += 1 - FIF1.compute_paths(X)
    FIF2 = FIF.FIForest(X2, time=times, innerproduct='auto', D='cosinus')
    un_FIF2 += 1 - FIF2.compute_paths(X)
    FIF3 = FIF.FIForest(X3, time=times, innerproduct='auto', D='cosinus')
    un_FIF3 += 1 - FIF3.compute_paths(X)
    FIF4 = FIF.FIForest(X4, time=times, innerproduct='auto', D='cosinus')
    un_FIF4 += 1 - FIF4.compute_paths(X)
    FIF5 = FIF.FIForest(X5, time=times, innerproduct='auto', D='cosinus')
    un_FIF5 += 1 - FIF5.compute_paths(X)

S = np.concatenate((un_Score0.reshape(-1,1), un_Score1.reshape(-1,1), un_Score2.reshape(-1,1),
 un_Score3.reshape(-1,1), un_Score4.reshape(-1,1), un_Score5.reshape(-1,1), un_FSDO0.reshape(-1,1),un_FSDO1.reshape(-1,1),
 un_FSDO2.reshape(-1,1),un_FSDO3.reshape(-1,1),un_FSDO4.reshape(-1,1),un_FSDO5.reshape(-1,1), un_FT0.reshape(-1,1), 
 un_FT1.reshape(-1,1), un_FT2.reshape(-1,1), un_FT3.reshape(-1,1), un_FT4.reshape(-1,1),un_FT5.reshape(-1,1),
  un_FIF0.reshape(-1,1), un_FIF1.reshape(-1,1), un_FIF2.reshape(-1,1), un_FIF3.reshape(-1,1), un_FIF4.reshape(-1,1), un_FIF5.reshape(-1,1)))

S = S / 10

with open("xp_robu_repetition10_2.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(S)


