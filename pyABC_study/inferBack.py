import os
import tempfile
from numpy import nan as NaN
import numpy as np
import pyabc
import matplotlib.pyplot as plt
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

print("Target data\n")
print(expData)


# Define prior distribution of parameters

paraPrior = pyabc.Distribution(
    lambdaN=pyabc.RV("uniform", 8, 15),
    kNB=pyabc.RV("uniform", -3, 2),
    muN=pyabc.RV("uniform", -5, 2),
    vNM=pyabc.RV("uniform", -7, 5),
    lambdaM=pyabc.RV("uniform", -3, 7),
    kMB=pyabc.RV("uniform", -5, 3),
    muM=pyabc.RV("uniform", -3, 4),
    sBN=pyabc.RV("uniform", -3, 2),
    iBM=pyabc.RV("uniform", -3, 2),
    muB=pyabc.RV("uniform", 0, 7),
    sAM=pyabc.RV("uniform", 3, 18),
    muA=pyabc.RV("uniform", 10, 60)
)

# Define ABC-SMC model

distance_adaptive = pyabc.AdaptivePNormDistance(p=2)

abc = pyabc.ABCSMC(models=solver.ode_model,
                   parameter_priors=paraPrior,
                   distance_function=euclidean_distance,
                   #distance_function=distance_adaptive,
                   population_size=1000,
                   eps=pyabc.MedianEpsilon(50, median_multiplier=1)
                   )

abc.new(db_path, expData)

history = abc.run(minimum_epsilon=0.1, max_nr_populations=25)

df, w = history.get_distribution(t=24)
pyabc.visualization.plot_kde_matrix(df, w)

plt.show()

pyabc.visualization.plot_acceptance_rates_trajectory(history)
plt.show()

pyabc.visualization.plot_epsilons(history)
plt.show()

# print(history.get_distribution(t=2))
#history.get_distribution(t=20)[0].to_csv(r"/home/yuan/wdnmd/MSc_Project/pyABC_study/outRaw.csv")

# Make a gif

limits = dict(
    lambdaN=(8, 15),
    kNB=(-3, 2),
    muN=(-5, 2),
    vNM=(-7, 5),
    lambdaM=(-3, 7),
    kMB=(-5, 3),
    muM=(-3, 4),
    sBN=(-3, 2),
    iBM=(-3, 2),
    muB=(0, 7),
    sAM=(3, 18),
    muA=(10, 60)
              )

df, w = history.get_distribution(t=24)
pyabc.visualization.plot_kde_matrix(df, w, limits=limits)

for tt in range(1,24):
    filename = r"/home/yuan/wdnmd/MSc_Project/pyABC_study/p1000e50t25/"+str(tt)+".png"
    df, w = history.get_distribution(t=tt)
    pyabc.visualization.plot_kde_matrix(df, w)
    plt.savefig(filename)