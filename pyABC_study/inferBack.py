import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import pyabc

from pyABC_study.ODE import ODESolver, euclidean_distance, PriorLimits
from pyABC_study.dataPlot import sim_data_plot, result_plot

# Get path

ROOT_DIR = os.path.abspath(os.curdir)
db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))


# Generate synthetic data

# True parameters

# paraInit = {
#     'iBM': 6.706790,
#     'kMB': 37.790301,
#     'kNB': 13.288773,
#     'lambdaM': 40.238402,
#     'lambdaN': 45.633238,
#     'muA': 39.136272,
#     'muB': 15.821665,
#     'muM': 34.883162,
#     'muN': 77.583389,
#     'sAM': 40.198178,
#     'sBN': 32.110228,
#     'vNM': 12.689222
# }

paraInit = {'iBM': 2.4041603100488587,
            'kMB': 0.14239564228380108,
            'kNB': 2.3405757296708396,
            'lambdaM': 1.9508302861494105,
            'lambdaN': 2.5284489000168113,
            'muA': 2.715326160638292,
            'muB': 0.35008723255144486,
            'muM': 0.1603505707251119,
            'muN': 2.2016772634585147,
            'sAM': 1.387525971337514,
            'sBN': 1.202190024316036,
            'vNM': 0.5119068430635925}

# Using default time points
solver = ODESolver()
expData = solver.ode_model(paraInit)

#normalise_data(expData)
print("Target data")
print(expData)


# Plot

sim_data_plot(solver.timePoint, expData)


# Define prior distribution of parameters
# Be careful that RV("uniform", -10, 15) means uniform distribution in [-10, 5], '15' here is the interval length

lim = PriorLimits(0, 50)

paraPrior = pyabc.Distribution(
    lambdaN=pyabc.RV("uniform", lim.lb, lim.interval_length),
    kNB=pyabc.RV("uniform", lim.lb, lim.interval_length),
    muN=pyabc.RV("uniform", lim.lb, lim.interval_length),
    vNM=pyabc.RV("uniform", lim.lb, lim.interval_length),
    lambdaM=pyabc.RV("uniform", lim.lb, lim.interval_length),
    kMB=pyabc.RV("uniform", lim.lb, lim.interval_length),
    muM=pyabc.RV("uniform", lim.lb, lim.interval_length),
    sBN=pyabc.RV("uniform", lim.lb, lim.interval_length),
    iBM=pyabc.RV("uniform", lim.lb, lim.interval_length),
    muB=pyabc.RV("uniform", lim.lb, lim.interval_length),
    sAM=pyabc.RV("uniform", lim.lb, lim.interval_length),
    muA=pyabc.RV("uniform", lim.lb, lim.interval_length)
)

# Define ABC-SMC model

#distance_adaptive = pyabc.AdaptivePNormDistance(p=2)

abc = pyabc.ABCSMC(models=solver.ode_model,
                   parameter_priors=paraPrior,
                   population_size=1000,
                   #distance_function=distance_adaptive,
                   distance_function=euclidean_distance,
                   eps=pyabc.MedianEpsilon(100, median_multiplier=1)
                   )

abc.new(db_path, expData)

max_population = 15

history = abc.run(minimum_epsilon=0.1, max_nr_populations=max_population)

result_plot(history, max_population)

