# **Physic Informed Neural Network**
# **This is the original documentation from Mishra and Molinaro, for the documentation for the master's thesis 'Error Estimation for Physics-Informed Neural Networks Approximating Semi-Linear Wave Equations' see the file README_SemeLinWave.md **
# **The original repository can be found here: https://github.com/mroberto166/PinnsSub**

Repository to reproduce the experiments in the papers :

   - Estimates on the generalization error of Physics Informed Neural Networks (PINNs) for approximating PDEs (https://arxiv.org/abs/2006.16144)
   - Estimates on the generalization error of Physics Informed Neural Networks (PINNs) for approximating PDEs II: A class of inverse problem (https://arxiv.org/abs/2007.01138)
   
The code is part of a bigger project that is under ongoing work. Therefore, there are commented lines of code that might confuse the reader.
The code focuses of the training of Physic Informed Neural Networks. It has been implemented to solve a large class of PDE. These are listed in the folder EquationModels. The corresponding equation has to be specified in the file ImportFile.py. The code run on both CPU and GPU.`  
## **Dependencies**

Dependecies are contained in the file requirement.txt and can be easily installed by typing in the terminal:

` python3 -m pip install -r requirements.txt ` 


## **Run the Code**
The ensemble training as described in "Deep learning observables in computational fluid dynamics" (https://arxiv.org/abs/1903.03040) can be run with

` python3 EnsembleTraining.py N_coll N_u N_int foldername cluster `

   - N_coll: Number of collocation points (where the PDE is enforced)
   - N_u: Number of initial and boundary points
   - N_int: Number of internal points (neither boundaries or initial points) where the solution of the PDE is known)
   - cluster: "true" or "false". Set it "false" unless a lsf-based cluster is available
   
Additional parameter have to be set in the code. The model in retrained 5 times for each hyperparameters configuration (parameter can be set in single_retraining.py)
The ensemble training can be run also for different number of training samples:

` python3 EnsembleTraining.py N_coll N_u N_int cluster `

Note:N_coll=n ** input_dimensions, N_u=(1+2*(input_dimensions-1)) * n ** (input_dimensions - 1), n\in\mathbb{N}

The convergence analysis can be run with:

` python3 SampleSensitivity.py foldername cluster `

This usually makes sense when uniformely distributed random points (point="random") are used. By default, the dataset is resampled 30 times and the model trained for each value of the number of points specified in the file SampleSensitivity.px and for each resampling of the datasets.

To get the error that we need 
By default

The code can be also run for a single configuration of the hyperaparmters and a single value of the number of training points with

` python3 PINNS2.py `

Parameters have to be specified in the corresponding file.

