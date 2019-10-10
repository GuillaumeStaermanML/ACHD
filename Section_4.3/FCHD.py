""" ACH depth

    Author : Guillaume Staerman
"""


"""The Area of the Convex Hull of Sampled curves (depth).

This is the implementation of The ACH depth which is a functional statistical depth measure for functional data.
"""



import numpy as np
from math import *
from itertools import combinations
from scipy.spatial import ConvexHull
from scipy.special import comb



class FuncCHD(object):
    """ Creates a functional convex hull depth.
    
    Attributes
    ----------
    X : Array-like (n_samples, p)
        Data used for training. p is the number of discretization points.
        
    times : Array
        The vector with the time values where functional data are observed. 
        
    J : int 
        The maximum value for which we average presents in the paper.
      
    Subsampling : booleen
        If False, we compute the exact depth defined in the paper. Otherwise,
        we compute only a subsample of all combinations of the U-statistics
        of different orders (J=n_sample for this).
        
    K : int
        subsampling size.
        
    Parrallelization : Booleen
        If true, an automatic parallelization is implemented to the exact 
        computation of the depth. Each U-statistics with fixed order is
        computed on each core.

    
    """
    
    
    def __init__(self,
                 X,
                 times,
                 J=2,
                 K=None,
                 Subsampling=False,
                 Mean_version=False):
        
        self.X = X
        self.times = times
        self.J = J
        self.subsampling = Subsampling
        self.Mean_version = Mean_version

        if (self.Mean_version == True):
    
            if (self.subsampling == True):

                self.J_Subsampling = J

                if K is None:
                    self.K = 5 * self.X.shape[0]
                else:
                    self.K = K

                self.V = [0] * self.K
                n0 = self.X.shape[0]
                p0 = self.X.shape[1]
                s = 0
                for k in range(self.K):
                    # On commence à 2 ici :
                    weight = np.zeros((self.J_Subsampling - 1))
                    for z in np.arange(2, self.J_Subsampling+1):
                        weight[z-2] = comb(n0,z)
                    weight  = weight / np.sum(weight)
                        
                    r = np.random.choice(np.arange(2, self.J_Subsampling+1), size=1)
                    subsample = np.random.choice(np.arange(n0), 
                                                 size=r, replace=False)

                    ## Mise en forme des données pour Qhull function :
                    Y = np.zeros((p0 * len(subsample), 2))
                    for l in range(len(subsample)):
                        Y[(p0 * (l)):(p0 * (l+1)) ,0] = self.times
                        Y[(p0 * (l)):(p0 * (l+1)) ,1] = self.X[subsample[l],:]
                    ################################################
                    #self.Y.append(Y)
                    hull = ConvexHull(Y) 
                    self.V[s] = [subsample, hull.volume]
                    s += 1
                    hull.close()


            else:

                self.V = [0] * (J-1)
               
                n0 = self.X.shape[0]
                p0 = self.X.shape[1]
                s = 0
                for j in range(2,J+1):
                    combi = list(combinations(np.arange(n0), j))

                    V1 = [0]* len(combi)
                    # On construit une matrice de dimension 2 (ou d+1) contenant les points 
                    # pour calculer l'enveloppe convexe :
                    k = 0

                    for j in combi:

                        # Compute the convex Hull of all combinations of the data.
                        Y = np.zeros((p0 * len(j), 2))
                        for l in range(len(j)):
                            Y[(p0*(l)):(p0 * (l+1)) ,0] = self.times
                            Y[(p0*(l)):(p0 * (l+1)) ,1] = X[j[l],:]
                        ################################################
                        #self.Y.append(Y)
                        hull = ConvexHull(Y) 
                        V1[k] = [j, hull.volume]
                        k += 1
                        hull.close()
                    self.V[s] = V1
                    s += 1
                            
        else:

            if (self.subsampling == True):

                self.J_Subsampling = J

                if K is None:
                    self.K = 5 * self.X.shape[0]
                else:
                    self.K = K

                self.V = [0] * self.K
                n0 = self.X.shape[0]
                p0 = self.X.shape[1]
                s = 0
                for k in range(self.K):
                    subsample = np.random.choice(np.arange(n0), 
                                                 size=self.J, replace=False)

                    ## Mise en forme des données pour Qhull function :
                    Y = np.zeros((p0 * len(subsample), 2))
                    for l in range(len(subsample)):
                        Y[(p0 * (l)):(p0 * (l+1)) ,0] = self.times
                        Y[(p0 * (l)):(p0 * (l+1)) ,1] = self.X[subsample[l],:]
                    ################################################
                    #self.Y.append(Y)
                    hull = ConvexHull(Y) 
                    self.V[s] = [subsample, hull.volume]
                    s += 1
                    hull.close()

            else:

                self.V = [0] * (J-1)
               
                n0 = self.X.shape[0]
                p0 = self.X.shape[1]
                s = 0
                
                combi = list(combinations(np.arange(n0), self.J))

                V1 = [0]* len(combi)
                # On construit une matrice de dimension 2 (ou d+1) contenant les points 
                # pour calculer l'enveloppe convexe :
                k = 0

                for j in combi:

                    # Compute the convex Hull of all combinations of the data.
                    Y = np.zeros((p0 * len(j), 2))
                    for l in range(len(j)):
                        Y[(p0*(l)):(p0 * (l+1)) ,0] = self.times
                        Y[(p0*(l)):(p0 * (l+1)) ,1] = X[j[l],:]
                    ################################################
                    #self.Y.append(Y)
                    hull = ConvexHull(Y) 
                    V1[k] = [j, hull.volume]
                    k += 1
                    hull.close()
                self.V[s] = V1
                s += 1
         
    def compute_depth(self, X_new=None):
        """
        compute_depth(X_new = None) 

        The depth of an input sample is computed w.r.t. 
        the sample X.
        
        Parameters
        ----------
        X_in : Array-like
                Data to be scored. 
        Returns
        -------
        float
        
            Depth score for a given data curve (or dataset).
        """
        if X_new is None:
            X_new = self.X
            
        # Ici, on calcule la profondeur pour certaines U-statistics choisies
        # dans l'initialisation de la classe.
        if (self.Mean_version == True):

            if (self.subsampling == True):
                p0 = X_new.shape[1]
                n0 = X_new.shape[0]
                Score = np.zeros((n0))

                for i in range(n0):
                    for j in range(len(self.V)):

                        Ytmp = np.vstack((self.times, X_new[i]))
                        Y = np.zeros((p0 * len(self.V[j][0]), 2))
                        for l in range(len(self.V[j][0])):
                            Y[(p0*(l)):(p0 * (l+1)) ,0] = self.times
                            Y[(p0*(l)):(p0 * (l+1)) ,1] = self.X[self.V[j][0][l],:]


                        hull = ConvexHull(np.concatenate((Y,Ytmp.T),axis=0))
                        Score[i] += self.V[j][1] / (1. * hull.volume) 
                        hull.close()

                    # Normalisation    
                    Score[i] = Score[i] / (1. * len(self.V))

                    
                    
                    
            else:
                n0 = X_new.shape[0]
                p0 = self.X.shape[1]
                n = self.X.shape[0]
                #p0 = X_new.shape[1]
                score = np.zeros((n0,self.J-1))    
                
                for i in range(n0):
                    for m in range(2,self.J+1):
                        combi = list(combinations(np.arange(n), m))
                        
                        k = 0
                        for j in combi:


                            Ytmp = np.vstack((self.times, X_new[i]))
                            Y = np.zeros((p0 * len(j), 2))
                            for l in range(len(j)):
                                Y[(p0*(l)):(p0 * (l+1)) ,0] = self.times
                                Y[(p0*(l)):(p0 * (l+1)) ,1] = self.X[j[l],:]

                   
                            hull = ConvexHull(np.concatenate((Y, Ytmp.T),axis=0))

                            score[i, m-2] += self.V[m-2][k][1] / (1. * hull.volume)
                            k += 1
                            hull.close()
                        score[i,m-2]  = score[i,m-2] / (1. * len(combi))


                # Normalisation 
                Score = np.mean(score,axis = 1)
        else:

            if (self.subsampling == True):
                p0 = X_new.shape[1]
                n0 = X_new.shape[0]
                Score = np.zeros((n0))

                for i in range(n0):
                    for j in range(len(self.V)):

                        Ytmp = np.vstack((self.times, X_new[i]))
                        Y = np.zeros((p0 * len(self.V[j][0]), 2))
                        for l in range(len(self.V[j][0])):
                            Y[(p0*(l)):(p0 * (l+1)) ,0] = self.times
                            Y[(p0*(l)):(p0 * (l+1)) ,1] = self.X[self.V[j][0][l],:]


                        hull = ConvexHull(np.concatenate((Y,Ytmp.T),axis=0))
                        Score[i] += self.V[j][1] / (1. * hull.volume) 
                        hull.close()

                    # Normalisation    
                    Score[i] = Score[i] / (1. * len(self.V))

            else:
                n0 = X_new.shape[0]
                p0 = self.X.shape[1]
                n = self.X.shape[0]
                #p0 = X_new.shape[1]
                score = np.zeros(n0)   
                
                for i in range(n0):
                    
                    combi = list(combinations(np.arange(n), self.J))
                    
                    k = 0
                    for j in combi:


                        Ytmp = np.vstack((self.times, X_new[i]))
                        Y = np.zeros((p0 * len(j), 2))
                        for l in range(len(j)):
                            Y[(p0*(l)):(p0 * (l+1)) ,0] = self.times
                            Y[(p0*(l)):(p0 * (l+1)) ,1] = self.X[j[l],:]

               
                        hull = ConvexHull(np.concatenate((Y, Ytmp.T),axis=0))

                        score[i] += self.V[0][k][1] / (1. * hull.volume)
                        k += 1
                        hull.close()
                    score[i]  = score[i] / (1. * len(combi))

               
                Score = score


                
        
        return Score
            
            