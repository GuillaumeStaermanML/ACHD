ACHD: The area of the convex hull of sampled curves: a robust functional statistical depth measure.
=========================================

This repository hosts Python code of the ACH depth algorithm: http://proceedings.mlr.press/v108/staerman20a.html.


Installation
------------
Download this repository and then run this python command in the folder:

.. code:: python

   python setup.py build_ext --inplace
   
Further, you can import the algorithm with the following command in your python script:

.. code:: python

   import achd as ACHD
   
   
   
   
Quick Start :
------------

Create a toy dataset :

.. code:: python

  import numpy as np 
  np.random.seed(42)
  m =100;n =100;tps = np.linspace(0,1,m);v = np.linspace(1,1.4,n)
  X = np.zeros((n,m))
  for i in range(n):
      X[i] = 30 * ((1-tps) ** v[i]) * tps ** v[i]
  Z1 = np.zeros((m))
  for j in range(m):
      if (tps[j]<0.2 or tps[j]>0.8):
          Z1[j] = 30 * ((1-tps[j]) ** 1.2) * tps[j] ** 1.2 
      else:
          Z1[j] = 30 * ((1-tps[j]) ** 1.2) * tps[j] ** 1.2 + np.random.normal(0,0.3,1)
  Z1[0] = 0
  Z1[m-1] = 0
  Z2 = 30 * ((1-tps) ** 1.6) * tps ** 1.6
  Z3 = np.zeros((m))
  for j in range(m):
      Z3[j] = 30 * ((1-tps[j]) ** 1.2) * tps[j] ** 1.2 + np.sin(2*np.pi*tps[j])

  Z4 = np.zeros((m))
  for j in range(m):
      Z4[j] = 30 * ((1-tps[j]) ** 1.2) * tps[j] ** 1.2

  for j in range(70,71):
      Z4[j] += 2

  Z5 = np.zeros((m))
  for j in range(m):
      Z5[j] = 30 * ((1-tps[j]) ** 1.2) * tps[j] ** 1.2 + 0.5*np.sin(10*np.pi*tps[j])

  X = np.concatenate((X,Z1.reshape(1,-1),Z2.reshape(1,-1),  
                       Z3.reshape(1,-1), Z4.reshape(1,-1), Z5.reshape(1,-1)), axis = 0)


   
And then use ACHD to rank functional dataset:

.. code:: python

   import achd as ACHD
   ACH = ACHD.ACHD(discretization_points=tps,combi_subsample=420, J_size=2)
   ACH.fit(X)
   Score = ACH.get_training_score()
   

If you want to use the fitted estimator use:

.. code:: python

   Score = ACH.decision_function(X_test)

   
Dependencies
------------

These are the dependencies to use ACHD:

* numpy 
* cython


Cite
----

If you use this code in your project, please cite::

   @InProceedings{pmlr-v108-staerman20a,
     title = 	 {The Area of the Convex Hull of Sampled Curves: a Robust Functional Statistical Depth measure},
     author =       {Staerman, Guillaume and Mozharovskyi, Pavlo and Cl\'emen{\c}on, St\'ephan},
     booktitle = 	 {Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics},
     pages = 	 {570--579},
     year = 	 {2020},
     volume = 	 {108},
     publisher =    {PMLR}
   }
