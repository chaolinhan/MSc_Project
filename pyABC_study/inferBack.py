import os
import pandas as pd
import tempfile
from numpy import nan as NaN
import numpy as np
import pyabc
import matplotlib.pyplot as plt
import copy

from pyABC_study.ODE import ODESolver, euclidean_distance, normalise_data

ROOT_DIR = os.path.abspath(os.curdir)
db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))


# Generate synthetic data

paraInit = {
    'iBM': 9.051270,
    'kMB': 40.881926,
    'kNB': 9.618762,
    'lambdaM': 41.405661,
    'lambdaN': 29.360990,
    'muA': 44.426018,
    'muB': 16.450285,
    'muM': 37.356256,
    'muN': 78.150011,
    'sAM': 33.580249,
    'sBN': 41.486109,
    'vNM': 13.005909
}

solver = ODESolver()
expData = solver.ode_model(paraInit)

#normalise_data(expData)

print("Target data")
print(expData)


# Define prior distribution of parameters
# Be careful that RV("uniform", -10, 15) means uniform distribution in [-10, 5], '15' here is the interval length

paraPrior = pyabc.Distribution(
    lambdaN=pyabc.RV("uniform", 0, 50),
    kNB=pyabc.RV("uniform", 0, 50),
    muN=pyabc.RV("uniform", 0, 50),
    vNM=pyabc.RV("uniform", 0, 50),
    lambdaM=pyabc.RV("uniform", 0, 50),
    kMB=pyabc.RV("uniform", 0, 50),
    muM=pyabc.RV("uniform", 0, 50),
    sBN=pyabc.RV("uniform", 0, 50),
    iBM=pyabc.RV("uniform", 0, 50),
    muB=pyabc.RV("uniform", 0, 50),
    sAM=pyabc.RV("uniform", 0, 50),
    muA=pyabc.RV("uniform", 0, 50)
)

# Define ABC-SMC model

#distance_adaptive = pyabc.AdaptivePNormDistance(p=2)

abc = pyabc.ABCSMC(models=solver.ode_model,
                   parameter_priors=paraPrior,
                   population_size=1000,
                   #distance_function=distance_adaptive,
                   distance_function=euclidean_distance,
                   eps=pyabc.MedianEpsilon(30, median_multiplier=1)
                   )

abc.new(db_path, expData)

max_population = 20

history = abc.run(minimum_epsilon=0.1, max_nr_populations=max_population)

df, w = history.get_distribution(t=max_population-1)
pyabc.visualization.plot_kde_matrix(df, w)

plt.show()

pyabc.visualization.plot_acceptance_rates_trajectory(history)
plt.show()

pyabc.visualization.plot_epsilons(history)
plt.show()

df.mean().to_csv( ROOT_DIR+"/pyABC_study/outSummary.csv")
df.std().to_csv( ROOT_DIR+"/pyABC_study/outSummStd.csv")
# print(history.get_distribution(t=2))
#history.get_distribution(t=20)[0].to_csv(r"/home/yuan/wdnmd/MSc_Project/pyABC_study/outRaw.csv")

# Make a gif

# limits = dict(
#     lambdaN=(-10, 15),
#     kNB=(-15, 2),
#     muN=(-15, 2),
#     vNM=(-23, 2),
#     lambdaM=(-3, 7),
#     kMB=(-15, 2),
#     muM=(-3, 4),
#     sBN=(-15, 2),
#     iBM=(-15, 2),
#     muB=(0, 12),
#     sAM=(-10, 30),
#     muA=(-20, 80)
#               )
#
#
# # TODO make file name dynamic
#
# for tt in range(1,max_population-1):
#     filename = ROOT_DIR+"/pyABC_study/plot/p500e50t25/"+str(tt)+".png"
#     df, w = history.get_distribution(t=tt)
#     pyabc.visualization.plot_kde_matrix(df, w, limits=limits)
#     plt.savefig(filename)


# Resume

# abc_continue = pyabc.ABCSMC(models=solver.ode_model,
#                    parameter_priors=paraPrior,
#                    distance_function=euclidean_distance,
#                    #distance_function=distance_adaptive,
#                    population_size=1000,
#                    eps=pyabc.MedianEpsilon(30, median_multiplier=1)
#                    )
#
# abc_continue.load(db_path, 3)