import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabc

from pyABC_study.ODE import ODESolver, euclidean_distance


# Read  and prepare raw data

rawData_path = os.path.abspath(os.curdir) + "/data/rawData.csv"
rawData = pd.read_csv(rawData_path).astype("float32")

timePoints = rawData.iloc[:, 0].to_numpy()

expData = rawData.iloc[:, 1:].to_dict(orient='list')
for k in expData:
    expData[k] = np.array(expData[k])
# normalise_data(expData)

print("Target data")
print(expData)


# Run a rough inference on the raw data

solver = ODESolver()
# Reload the timepoints to be calculated in ODE solver
solver.timePoint = timePoints

paraPrior = pyabc.Distribution(
    lambdaN=pyabc.RV("uniform", 0, 100),
    kNB=pyabc.RV("uniform", 0, 100),
    muN=pyabc.RV("uniform", 0, 100),
    vNM=pyabc.RV("uniform", 0, 100),
    lambdaM=pyabc.RV("uniform", 0, 100),
    kMB=pyabc.RV("uniform", 0, 100),
    muM=pyabc.RV("uniform", 0, 100),
    sBN=pyabc.RV("uniform", 0, 100),
    iBM=pyabc.RV("uniform", 0, 100),
    muB=pyabc.RV("uniform", 0, 100),
    sAM=pyabc.RV("uniform", 0, 100),
    muA=pyabc.RV("uniform", 0, 100)
)

abc = pyabc.ABCSMC(models=solver.ode_model,
                   parameter_priors=paraPrior,
                   distance_function=euclidean_distance,
                   # distance_function=distance_adaptive,
                   population_size=300,
                   eps=pyabc.MedianEpsilon(100, median_multiplier=1)
                   )

db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))
abc.new(db_path, expData)

max_population = 21

history = abc.run(minimum_epsilon=0.1, max_nr_populations=max_population)


# Plot the results

pyabc.visualization.plot_acceptance_rates_trajectory(history)
plt.show()

pyabc.visualization.plot_epsilons(history)
plt.show()

df, w = history.get_distribution(t=max_population - 1)

np.argmax()

pyabc.visualization.plot_kde_matrix(df, w)
plt.show()


# Print results

# Mean of last population
print(df.mean())

# Sum of weight * particles
for i in range(12):
    print(df.iloc[:, i].name + '\t\t%.6f' % (df.iloc[:, i] * w).sum())

# Particle with maximal weight
print(df.iloc[w.argmax(),:])

"""
Output from one run:


mean() method: 

iBM         9.737715
kMB        47.458632
kNB         9.973562
lambdaM    39.107247
lambdaN    25.262527
muA        47.041885
muB        15.762823
muM        39.539603
muN        79.994190
sAM        38.563204
sBN        37.618288
vNM        13.229111


df*w sum() method:

iBM		9.051270
kMB		40.881926
kNB		9.618762
lambdaM		41.405661
lambdaN		29.360990
muA		44.426018
muB		16.450285
muM		37.356256
muN		78.150011
sAM		33.580249
sBN		41.486109
vNM		13.005909

maximal weight method:
iBM         6.706790
kMB        37.790301
kNB        13.288773
lambdaM    40.238402
lambdaN    45.633238
muA        39.136272
muB        15.821665
muM        34.883162
muN        77.583389
sAM        40.198178
sBN        32.110228
vNM        12.689222
"""


# Least squares using LM

def residual(para, ydata):
    simulationData = solver.ode_model(para)
    return euclidean_distance(ydata, simulationData)