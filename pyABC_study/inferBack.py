import os
import tempfile
from numpy import nan as NaN
import numpy as np
import pyabc
import copy

from pyABC_study.ODE import ODESolver, euclidean_distance, normalise_data

db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))


# Generate synthetic normalised data

paraInit = {"lambdaN": 13.753031, "kNB": 1.581684, "muN": -0.155420, "vNM": 0.262360,
            "lambdaM": 2.993589, "kMB": 0.041040, "muM": 0.201963,
            "sBN": 1.553020, "iBM": -0.046259, "muB": 1.905163,
            "sAM": 11.001731, "muA": 23.022678}

solver = ODESolver()
expData = solver.ode_model(paraInit)
normalise_data(expData)


# Define prior distribution of parameters

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

abc = pyabc.ABCSMC(models=solver.ode_model,
                   parameter_priors=paraPrior,
                   distance_function=euclidean_distance,
                   population_size=100,
                   eps=pyabc.MedianEpsilon(500, median_multiplier=1)
                   )

abc.new(db_path, expData)

history = abc.run(minimum_epsilon=10, max_nr_populations=20)

# print(history.get_distribution(t=2))
#history.get_distribution(t=20)[0].to_csv(r"/Users/chaolinhan/OneDrive/PostgraduateProject/pyABC_study/outRaw.csv")