cdef extern from "achd.hxx":
    cdef cppclass ACHD:
        ACHD (int, int, int)
        void compute_volume_train(double*, double*, double*, int*, int, int) 
        void compute_volume_test(double*, double*, double*, int*, int, int, double*) 