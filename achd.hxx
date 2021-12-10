#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <unordered_set>

#define RANDOM_ENGINE std::mt19937_64
#define RANDOM_SEED_GENERATOR std::random_device


/****************************
        Class ACHD
 ****************************/

class ACHD
{
    private:
        int J_size;
        int combi_subsample;
        unsigned random_seed;
    protected:

    public:
        ACHD (int, int, int);
        ~ACHD ();
        void compute_volume_train(double*, double*, double*, int*, int, int);
        void compute_volume_test(double*, double*, double*, int*, int, int, double*);
};




/********************************
        Utility functions
 ********************************/
inline std::vector<int> sample_without_replacement (int, int, RANDOM_ENGINE&);

