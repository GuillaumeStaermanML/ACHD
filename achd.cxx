#include "achd.hxx"
extern "C"{
#include "libqhull.h"
#include "qhull_a.h"
}
#include "qhAdapter.h"


/********************************
	Utility functions
 ********************************/
inline std::vector<int> sample_without_replacement (int k, int N, RANDOM_ENGINE& gen)
	/* Sample k elements from the range [0, N-1] without replacement  */
{

    // Create an unordered set to store the samples
    std::unordered_set<int> samples;

    // Sample and insert values into samples
    for (int r=N-k; r<N; ++r)
    {
        int v = std::uniform_int_distribution<>(0, r)(gen);
        if (!samples.insert(v).second) samples.insert(r);
    }

    // Copy samples into vector
    std::vector<int> result(samples.begin(), samples.end());

    // Shuffle vector
    std::shuffle(result.begin(), result.end(), gen);

    return result;
}

/****************************
        Class ACHD
 ****************************/


ACHD::ACHD (int J_size_in, int combi_subsample_in, int random_seed_in=-1)
{
  J_size = J_size_in;
  combi_subsample = combi_subsample_in;
  if (random_seed_in < 0) 
  {
     RANDOM_SEED_GENERATOR random_seed_generator;
     random_seed = random_seed_generator();
  } 
  else 
  {
     random_seed = (unsigned) random_seed_in;
  }
}

ACHD::~ACHD ()
{}


void ACHD::compute_volume_train(double* S,  
                                double* X,  
                                double* vol_subsample, 
                                int* idx_subsample,  
                                int n_samples,  
                                int discretization_size)
{
  std::vector<double> X_subsample (2 * (J_size + 1) * discretization_size, 0.0);
  double vol_num=0.0;
  double vol_denum=1.0;
  int size_conv_num = discretization_size * J_size;
  int size_conv_denum =  discretization_size * (J_size+1);
  for (int k=0; k<combi_subsample; k++)
  { 
      RANDOM_ENGINE random_engine (random_seed+k);

      // Draw a subsample of size J_size
      std::vector<int> subsample_index = sample_without_replacement (J_size, n_samples, random_engine);

      for (int i=0; i<J_size; i++) 
      {   
          idx_subsample[k * J_size + i] = subsample_index[i];
          for (int j=0; j<2*discretization_size; j++)
          {
              X_subsample[j + 2 * discretization_size * i] = X[subsample_index[i] * 2 * discretization_size + j];
          }
      }

      // Compute the volume of the convex hull of the subsample
      vol_subsample[k] = convvol(&X_subsample[0], size_conv_num, 2);

      // Compute the volume of the convex hull of the subsample concatenate with each sample separately
      for (int i=0; i<n_samples; i++)
      { 
          int index = i * 2 * discretization_size;

          // Add X[index] to the Subsample
          for (int j=0; j<2 * discretization_size; j++) 
          {
              X_subsample[j + J_size * discretization_size * 2] = X[i * 2 * discretization_size +j]; 
          }

          vol_denum = convvol(&X_subsample[0], size_conv_denum , 2);
          S[i] +=   vol_subsample[k] / vol_denum;
      }
  }
}

void ACHD::compute_volume_test(double* S_new, 
                               double* X,  
                               double* vol_subsample, 
                               int* idx_subsample,  
                               int discretization_size, 
                               int n_samples_new=NULL, 
                               double* X_new=NULL)
{	
  if (X_new == NULL)
    {}
  else
  { 
    std::vector<double> X_subsample (2 * (J_size + 1) * discretization_size, 0.0);
    double vol_denum = 1.0;
    int size_conv_denum =  discretization_size * (J_size+1);

    for (int k=0; k<combi_subsample; k++)
    {
      for (int i=0; i<J_size; i++) 
      {   
          int idx = idx_subsample[k * J_size+i];
          for (int j=0; j<2*discretization_size; j++)
          { 
              X_subsample[j + 2 * discretization_size * i] = X[idx * 2 * discretization_size + j];
          }
      }

      for (int i=0; i<n_samples_new; i++)
      {   
          int index = i * 2 * discretization_size;

          for (int j=0; j<2 * discretization_size; j++) 
          {
              X_subsample[j + J_size * discretization_size * 2] = X_new[i * 2 * discretization_size +j];
          } 

          vol_denum = convvol(&X_subsample[0], size_conv_denum , 2);
          S_new[i] +=   vol_subsample[k] / vol_denum;
      }
    }
  }
}






























