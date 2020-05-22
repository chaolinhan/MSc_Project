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


# Generate synthetic normalised data

paraInit = {"lambdaN": 13.753031, "kNB": 1.581684, "muN": -0.155420, "vNM": 0.262360,
            "lambdaM": 2.993589, "kMB": 0.041040, "muM": 0.201963,
            "sBN": 1.553020, "iBM": -0.046259, "muB": 1.905163,
            "sAM": 11.001731, "muA": 23.022678}

solver = ODESolver()
expData = solver.ode_model(paraInit)

###
rawData_path = ROOT_DIR + "/data/rawData.csv"
rawData = pd.read_csv(rawData_path, header=None)
timePoints = rawData.iloc[:,0].to_numpy()
rawDataArray = rawData.iloc[:,1:].transpose().to_numpy()
rawData = rawData.rename(columns={0:"time", 1:"N", 2:"M", 3:"B", 4:"A"})
rawDataDict = rawData.iloc[:,1:].to_dict(orient = 'list')

expData = rawDataDict
###

normalise_data(expData)

print("Target data")
print(expData)


# Define prior distribution of parameters
# Be careful that RV("uniform", -10, 15) means uniform distribution in [-10, 5], '15' here is the interval length

paraPrior = pyabc.Distribution(
    lambdaN=pyabc.RV("uniform", 10, 6),
    kNB=pyabc.RV("uniform", 1, 1),
    muN=pyabc.RV("uniform", -1, 2),
    vNM=pyabc.RV("uniform", -1, 2),
    lambdaM=pyabc.RV("uniform", 2, 2),
    kMB=pyabc.RV("uniform", -1, 2),
    muM=pyabc.RV("uniform", -1, 2),
    sBN=pyabc.RV("uniform", 1, 1),
    iBM=pyabc.RV("uniform", -1, 2),
    muB=pyabc.RV("uniform", 1, 1.5),
    sAM=pyabc.RV("uniform", 10, 3),
    muA=pyabc.RV("uniform", 20, 6)
)

# Define ABC-SMC model

distance_adaptive = pyabc.AdaptivePNormDistance(p=2)

abc = pyabc.ABCSMC(models=solver.ode_model,
                   parameter_priors=paraPrior,
                   distance_function=euclidean_distance,
                   #distance_function=distance_adaptive,
                   population_size=1000,
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