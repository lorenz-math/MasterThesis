import os
import sys
import itertools
import subprocess
from ImportFile import *

###############################################
# Convergence Analysis for PINNS, see README_SemilinWave for description of parameters
N_int = [0]
N_coll = [64,343,1000,2197,4096,8000,15625]
N_u=[80,245,500,845,1280,2000,3125]


N_coll = np.array(N_coll)
N_u=np.array(N_u)
N_int=np.array(N_int)
###############################################
# See file EnsambleTraining for the description of the parameters.
"""
Creation of NN with 2 hidden layers a 80 neurons, tanh activation function, full batch training
and midpoint rule ("quad") for selection of training points; Most other parameters like
regularization_parameter, residual_parameter and kernel_regularizer can be ignored for the 
semilinear wave function
"""
n_time_steps = 0
n_object = 0
ob = "None"
time_dimensions = 1
parameter_dimensions = 0
n_out = 1
folder_name = sys.argv[1]
point = "quad"
validation_size = 0.0
network_properties = {
    "hidden_layers": 2,
    "neurons": 80,
    "residual_parameter": 1,
    "kernel_regularizer": 2,
    "regularization_parameter": 0,
    "batch_size": "full",
    "epochs": 1,
    "activation": "tanh",
}
shuffle = "false"
cluster = sys.argv[2]

if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
settings = list(itertools.product(N_coll, N_u, N_int))


# for setup in settings:
for i in range(N_coll.shape[0]):

    '''N_coll_set = setup[0]
    N_u_set = setup[1]
    N_int_set = setup[2]'''
    N_coll_set = N_coll[i]
    N_u_set = N_u[i]
    N_int_set = N_int[0]

    print("\n")
    print("##########################################")
    print("Number of samples:")
    print(" - Collocation points:", N_coll_set)
    print(" - Initial and boundary points:", N_u_set)
    print(" - Internal points:", N_int_set)
    print("\n")
    folder_path = os.path.join(folder_name, str(int(N_u_set)) + "_" + str(int(N_coll_set)) + "_" + str(int(N_int_set)))

    arguments = list()

    arguments.append(str(N_coll_set))
    arguments.append(str(N_u_set))
    arguments.append(str(N_int_set))
    arguments.append(str(n_time_steps))
    arguments.append(str(n_object))
    arguments.append(str(ob))
    arguments.append(str(time_dimensions))
    arguments.append(str(parameter_dimensions))
    arguments.append(str(n_out))

    arguments.append(str(folder_path))
    arguments.append(str(point))
    arguments.append(str(validation_size))
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        arguments.append("\'" + str(network_properties).replace("\'", "\"") + "\'")
    else:
        arguments.append("\""+str(network_properties).replace("\"", "\'")+"\"")

    arguments.append(str(shuffle))
    arguments.append(str(cluster))

    if cluster == "true":
        string_to_exec = "bsub -W 1:00 python3 single_training.py "
    else:
        string_to_exec = "python single_training.py "
    for arg in arguments:
        string_to_exec = string_to_exec + " " + str(arg)
    os.system(string_to_exec)

