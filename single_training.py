import os
import sys
import json
import random
import subprocess

random.seed(42)

N_coll = int(sys.argv[1])
N_u = int(sys.argv[2])
N_int = int(sys.argv[3])
n_time_steps = int(sys.argv[4])
n_object = int(sys.argv[5])
ob = sys.argv[6]
time_dimensions = int(sys.argv[7])
parameter_dimensions = int(sys.argv[8])
n_out = int(sys.argv[9])
folder_path = sys.argv[10]
point = sys.argv[11]
validation_size = float(sys.argv[12])
network_properties = json.loads(sys.argv[13].replace("\'","\""))
shuffle = sys.argv[14]
cluster = sys.argv[15]

if point == "sobol":
    skip = max(N_u, N_coll) * validation_size
elif point == "random":
    skip = 42
elif point == "quad":
    skip = 0
else:
    raise ValueError()


os.mkdir(folder_path)
for i in range(1):
    if point == "sobol":
        rs = int(skip * i)
    elif point == "random":
        rs = i
        print(rs)
    #rs is random_seed for randomized sampling, not used when using midpoint rule
    elif point == "quad":
        rs = 1
    else:
        raise ValueError()

    
    folder_path_sample = os.path.join(folder_path, "Sample_"+ str(i))

    arguments = list()
    arguments.append(str(rs))
    arguments.append(str(N_coll))
    arguments.append(str(N_u))
    arguments.append(str(N_int))
    arguments.append(str(n_time_steps))
    arguments.append(str(n_object))
    arguments.append(str(ob))
    arguments.append(str(time_dimensions))
    arguments.append(str(parameter_dimensions))
    arguments.append(str(n_out))
    arguments.append(str(folder_path_sample))
    arguments.append(str(point))
    arguments.append(str(validation_size))
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        arguments.append("\'" + str(network_properties).replace("\'", "\"") + "\'")
    else:
        arguments.append("\"" + str(network_properties).replace("\"", "\'") + "\"")
    arguments.append(str(shuffle))
    arguments.append(str(cluster))

    if cluster == "true":
        string_to_exec = "bsub -W 00:10 python3 single_retraining.py "
    else:
        string_to_exec = "python single_retraining.py "
    for arg in arguments:
        string_to_exec = string_to_exec + " " + arg
    print(string_to_exec)
    os.system(string_to_exec)
