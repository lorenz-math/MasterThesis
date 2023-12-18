This is the code for the numerical experiments in the master's thesis 'Error Estimation for Physics-Informed Neural Networks Approximating Semi-Linear Wave Equations' by Meike Beatrice Lorenz. The code is based on S. Mishra and R. Molinaro's code for the papers 'Estimates on the generalization error of Physics Informed Neural Networks (PINNs) for approximating PDEs' (https://arxiv.org/abs/2006.16144) and 'Estimates on the generalization error of Physics Informed Neural Networks (PINNs) for approximating PDEs II: A class of inverse problem' (https://arxiv.org/abs/2007.01138) but certain parts were changed to fit the specifics of the semi-linear wave equation. 

As a side note, it is not necessary to install all the requirements in requirements.txt. The most important package to have is a Pytorch version <2.0.

The following changes were made to Mishra and Molinaro's code:
The files SemiLinWave.py, PlotConvAnalysis_SemiLinWave.py, CollectEnsembleSemiLineWave.py and CollectUtils_SemiLinWave.py are new and written by me. The actual solution and boundary and initial data is encoded in the new file SemiLinWave.py as well as methods to compute the PDE residual and the generaliztation error. PlotConvAnalysis_SemiLinWave.py plots the generalization error, training error and bound, based on data that is collected in CollectEnsembleSemiLineWave.py and CollectUtils_SemiLinWave.py. All of these files are new but similar in architecture to the original PlotConvAnalysis.py, CollectUtils.py and CollectEnsembleData.py. When SampleSensitivity.py is called it trains multiple PINNs for each number of training points via calling single_training.py, then single_retraining.py and lastly PINNS2.py. In SampleSensitivity.py the number of collocations points within the domain is the variable 'N_coll' and the number of training points for boundary and initial data together is specified as 'N_u'. 'N_u' is later divided into training points for the boundary data and for the initial data with ('N_u' / (1+2* (input_dimensions-1))) training points for initial data and ('N_u' / (1+2* (input_dimensions-1))) * (2 * (input_dimensions-1)) training points for the boundary data, where input_dimension is the sum of the space and time dimensions (in the example in the thesis, this is 3). 'N_int' is disregarded and set to zero as it specifies the nuber of internal boundary points where the function value is known for the inverse problem. The trained PINNs and their information is saved and later used in CollectEnsembleSemiLinWave.py and CollectUtils_SemiLinWave.py to compute the training error and the constants for the bound.

Within Mishra and Molinaro's original files I added the method VeryCustomLoss in ModelClassTorch2.py that modifies the loss function of the neural network to one that uses the additional residual terms that arise due to the second derivative in the semilinear wave equations and are necessitated by the proof of Theorem 3.3 in my thesis. Moreover, in DatasetTorch2.py I added the option "quad" to the method generator_points to generate training points according to the midpoint rule. Lastly, there have been tiny changes throughout to integrate VeryCustomLoss and the "quad" option properly into the architecture.

I have left the rest of the code largely untouched with the exception of deleting parts that were completely irrelevant to my thesis. This is why in some parts of this code there might still be some additional options that don't seem to make much sense but I did not want to delete them in case there are some dependencies that would break the code otherwise, plus some of them are used for parameters that are not implemented in this very basic version of a physics-informed neural networks but could be helpful in fine-tuning the PINN in the future.

I have used this code to compare the generalization and training error to the theoretical bound I found analytically. This culminates in Fig 1 on page 47. To create this graph first run

` python SampleSensitivity.py foldername False`.

Here one has to note that in SampleSensitivity.py 'N_coll' must be n ** (d+1) and 'N_u' must be 5 * m ** d for n and m in the natural numbers, and d the spatial dimensions (here d =3); foldername is the name of the folder that the models are saved at.

Then run

` CollectEnsembleSemiLinWave.py `

and lastly run

` PlotConvAnalysis_SemiLinWave.py `.

Make sure to change the folder names in CollectEnsembleSemiLinWave.py and PlotConvAnalysis_SemiLinWave.py to foldername used earlier.

The data that is the bases for figure 1 in the thesis can be found in the folder 'Run05_fin'.