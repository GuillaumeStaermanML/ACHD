import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pandas as pd
from multiprocessing import Pool
from functools import partial
from itertools import combinations
import csv
import warnings
warnings.filterwarnings("ignore")
import FCHD
import os
from sklearn.metrics import auc, roc_curve, roc_auc_score


X = pd.read_csv('Dataset_brownian_geom.csv', header=None)
X = X.as_matrix()


Y = pd.read_csv('Dataset_deux.csv', header=None)
Y = Y.as_matrix()

times = np.linspace(0, 1, X.shape[1])




# Dataset 1:

FCHD1 = FCHD.FuncCHD(X, times, J=2)
Score1 = FCHD1.compute_depth(X)

FCHD2 = FCHD.FuncCHD(X,times, J=3)
Score2 = FCHD2.compute_depth(X)

FCHD3 = FCHD.FuncCHD(X,times, J=4)
Score3 = FCHD3.compute_depth(X)

# Dataset 2:

FCHD4 = FCHD.FuncCHD(Y,times, J=2)
Score4 = FCHD4.compute_depth(Y)

FCHD5 = FCHD.FuncCHD(Y,times, J=3)
Score5 = FCHD5.compute_depth(Y)

FCHD6 = FCHD.FuncCHD(Y,times, J=4)
Score6 = FCHD6.compute_depth(Y)




Score = np.concatenate((Score1.reshape(1,-1), Score2.reshape(1,-1), Score3.reshape(1,-1)
	, Score4.reshape(1,-1), Score5.reshape(1,-1), Score6.reshape(1,-1)))



with open("Rankrank_score.csv", "w") as f_write:
    writer = csv.writer(f_write)
    writer.writerows(Score)