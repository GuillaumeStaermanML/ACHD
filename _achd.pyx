# Cython wrapper for the Area of the Convex hull of Sampled curves depth (ACHD)


# distutils: language = C++
# distutils: sources  = achd.cxx
# cython: language_level = 3

import cython
import numpy as np
cimport numpy as np
from version import __version__

cimport __achd

np.import_array()

cdef class ACHD:
    """
    Computes the Area of the Convex Hull of sampled curves. 
    This implementation return 1-D where D is the Data Depth 
    defined in the paper in order to define an anomaly score.


    References:
        Staerman,G., Mozharovskyi, P., Clémençon, S., d'Alché-Buc. (2020). 
        The Area of the Convex Hull of Sampled Curves: a Robust Statistical
         Functional Depth measure. AISTATS 2020.

    Parameters
    ----------
    discretization_points: array of shape (discretization_size)
                           Measurements of the time series.
    J_size: int, default=2.
            The size of the sampled subsample where convex hulls are computed.
            It corresponds to the number J of the original paper

    combi_subsample: int, default=100.
                     The number of combinations of subsamples that are drawn in a uniform way
                     from the data. It corresponds to the number K of the original paper.
    
    References
    ----------

    ..  [1] Staerman,G., Mozharovskyi, P., Clémençon, S., d'Alché-Buc. (2020). 
        The Area of the Convex Hull of Sampled Curves: a Robust Statistical
         Functional Depth measure. AISTATS 2020.
    """
    cdef int n_samples;
    cdef int discretization_size;
    cdef int J_size;
    cdef public int combi_size;
    cdef double [:] discretization_points;
    cdef double [:] score_samples;
    cdef int [:] idx_subsamples;
    cdef double [:] vol_subsamples;
    cdef double [:,:] data_rearranged;

    cdef __achd.ACHD* thisptr

    @cython.boundscheck(False)
    @cython.wraparound(False)

    def __cinit__ (self,
                   np.ndarray[double, ndim=1] discretization_points not None,
                   int J_size=2, 
                   int combi_subsample=100, 
                   int seed=-1):

        # Initialize the class defined in C++:
        self.thisptr = new __achd.ACHD (J_size, combi_subsample, seed)


        self.combi_size = combi_subsample
        self.J_size = J_size

        if not discretization_points.flags['C_CONTIGUOUS']:
            discretization_points = discretization_points.copy(order='C')

        self.discretization_points = discretization_points

        if (self.combi_size < 1):
            raise Exception("The number of subsample combinations must be greater than 0")
        if (self.J_size < 2):
            raise Exception("The subsample size must be greater than or equal to two")

    @cython.boundscheck(False)
    @cython.wraparound(False) 
    def fit(self, np.ndarray[double, ndim=2] X not None):
        """
        Fit estimator.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, discretization_size)
            The input samples. 

        Returns
        -------
        self : object
            Fitted estimator.
        """
        cdef np.ndarray[double, ndim=1, mode="c"] S
        cdef np.ndarray[double, ndim=2, mode="c"] X_rearranged
        cdef np.ndarray[double, ndim=1, mode="c"] vol_subsamples
        cdef np.ndarray[int, ndim=1, mode="c"] idx_subsamples

        self.n_samples = X.shape[0]
        self.discretization_size  = X.shape[1]


        if (self.discretization_size != len(np.asarray(self.discretization_points))):
            raise Exception("Discretization size must be  equal to the number of data measurements")

        S = np.zeros(self.n_samples, dtype=np.float64, order='C')
        vol_subsamples = np.zeros(self.combi_size , dtype=np.float64, order='C')
        idx_subsamples = np.zeros(self.combi_size * self.J_size , dtype=np.int32, order='C')
        X_rearranged = np.zeros((self.n_samples, 2*self.discretization_size), dtype=np.float64, order='C')
        discretization_repeated = np.repeat(np.asarray(self.discretization_points).reshape(1,-1), self.n_samples, axis=0)
        X_rearranged[:, ::2] = discretization_repeated
        X_rearranged[:,1::2] = X

        self.thisptr.compute_volume_train(<double*> np.PyArray_DATA(S),
                                          <double*> np.PyArray_DATA(X_rearranged),
                                          <double*> np.PyArray_DATA(vol_subsamples), 
                                          <int*> np.PyArray_DATA(idx_subsamples), 
                                          self.n_samples, 
                                          self.discretization_size)
        
        self.score_samples = 1 - (S / self.combi_size)
        self.idx_subsamples = idx_subsamples
        self.vol_subsamples = vol_subsamples
        self.data_rearranged = X_rearranged

        return 0


    def get_training_score(self):
        """
         the anomaly score of the training sample. It is defined as one minus the depth
         function defined in the paper.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal.
        """
        return np.asarray(self.score_samples)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)      
    def decision_function(self,  np.ndarray[double, ndim=2] X_new=None):
        """
        Anomaly score of X_new based on the fitted estimator.

        Parameters
        ----------
        X : {array-like} of shape (n_samples_new, discretization_size)
            The new/test samples. 

        Returns
        -------
        scores : ndarray of shape (n_samples_new,)
            The anomaly score of the input samples.
            The lower, the more abnormal. 
        """
        cdef np.ndarray[double, ndim=1, mode="c"] S
        cdef np.ndarray[double, ndim=2, mode="c"] X_new_rearranged
     
        if X_new is None:
            return self.get_training_score()        
        else:
            if not X_new.flags['C_CONTIGUOUS']:
                X_new = X_new.copy(order='C')
            n_samples_new = X_new.shape[0]

            if (self.discretization_size != n_samples_new):
                raise Exception("Data measurements of X_new must be equal to those of X")

            S = np.zeros(n_samples_new, dtype=np.float64, order='C')
            X_new_rearranged = np.zeros((n_samples_new, 2*self.discretization_size), dtype=np.float64, order='C')
            discretization_repeated_new = np.repeat(np.asarray(self.discretization_points).reshape(1,-1), n_samples_new, axis=0)
            X_new_rearranged[:, ::2] = discretization_repeated_new
            X_new_rearranged[:,1::2] = X_new

            self.thisptr.compute_volume_test(<double*> np.PyArray_DATA(S), 
                                             <double*> np.PyArray_DATA(np.asarray(self.data_rearranged)),
                                             <double*> np.PyArray_DATA(np.asarray(self.vol_subsamples)),
                                             <int*> np.PyArray_DATA(np.asarray(self.idx_subsamples)), 
                                             self.discretization_size, 
                                             n_samples_new, 
                                             <double*> np.PyArray_DATA(X_new_rearranged))

            S = 1 - (S / self.combi_size)

        return S


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit_predict(self, double contamination_level=0.1):
        """Returns labels for the training dataset with a fixed contamination level.
        Returns 1 for outliers and 0 for inliers.
        Parameters
        ----------
        contamination_level : float between 0 and 1.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            0 for inliers, 1 for outliers.
        """
        y_pred = np.zeros(self.n_samples) + 1
        num_temp = int((1 - contamination_level) * self.n_samples)
        y_pred[np.argsort(self.get_training_score())[:num_temp]] = 0

        return y_pred

    @cython.boundscheck(False)
    @cython.wraparound(False)    
    def predict(self, np.ndarray[double, ndim=2] X_new=None, double contamination_level=0.1):
        """Returns labels for the training dataset with a fixed contamination level.
        Returns 1 for outliers and 0 for inliers.
        Parameters
        ----------
        X_new : {array-like} of shape (n_samples_new, discretization_size)
                New/test samples.

        contamination_level : float between 0 and 1.
        Returns
        -------
        y_pred : ndarray of shape (n_samples_new,)
            0 for inliers, 1 for outliers.
        """

        if X_new is None:
            return self.fit_predict()
        else:
            n_samples_new = X_new.shape[0]
            Score = self.decision_function(X_new)
            y_pred = np.zeros(n_samples_new) + 1
            num_temp = int((1 - contamination_level) * n_samples_new)           
            y_pred[np.argsort(Score)[:num_temp]] = 0

            return y_pred


