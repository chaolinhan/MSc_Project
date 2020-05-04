import os
import tempfile
from numpy import nan as NaN
import numpy as np
import pyabc
import copy

from pyABC_study.ODE import ODESolver, euclidean_distance

db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))


# Test ODE solver
paraInit = {"lambdaN": 13.753031, "kNB": 1.581684, "muN": -0.155420, "vNM": 0.262360,
            "lambdaM": 2.993589, "kMB": 0.041040, "muM": 0.201963,
            "sBN": 1.553020, "iBM": -0.046259, "muB": 1.905163,
            "sAM": 11.001731, "muA": 23.022678}
#
# solver = ODESolver()
# ode_test = solver.ode_model(paraInit)
# ode_test


# Generate synthetic data (normalised)

solver = ODESolver()

expData = solver.ode_model(paraInit)
for key in expData:
    expData[key] = (expData[key] - expData[key].mean()) / expData[key].std()



# Test distance function

newData = copy.deepcopy(expData)
euclidean_distance(expData, newData, normalise=True)
# newData["N"][2] = 100
# print(distance(newData, expData))


# Define prior distribution od parameters

paraPrior = pyabc.Distribution(
    lambdaN=pyabc.RV("uniform", 10, 15),
    kNB=pyabc.RV("uniform", 0, 3),
    muN=pyabc.RV("uniform", -1, 1),
    vNM=pyabc.RV("uniform", -1, 1),
    lambdaM=pyabc.RV("uniform", 0, 5),
    kMB=pyabc.RV("uniform", -1, 1),
    muM=pyabc.RV("uniform", -1, 1),
    sBN=pyabc.RV("uniform", 0, 3),
    iBM=pyabc.RV("uniform", -1, 1),
    muB=pyabc.RV("uniform", -1, 5),
    sAM=pyabc.RV("uniform", 8, 15),
    muA=pyabc.RV("uniform", 20, 25)
)

# Define ABC-SMC model

abc = pyabc.ABCSMC(models=ODEmodel,
                   parameter_priors=paraPrior,
                   distance_function=distance,
                   population_size=50,
                   # transitions=pyabc.LocalTransition(k_fraction=.3)
                   eps=pyabc.MedianEpsilon(500, median_multiplier=0.7)
                   )

abc.new(db_path, expData)

history = abc.run(minimum_epsilon=1, max_nr_populations=4)

# print(history.get_distribution(t=2))
history.get_distribution(t=3)[0].to_csv(r"/Users/chaolinhan/OneDrive/PostgraduateProject/pyABC_study/pop3outRaw.csv")