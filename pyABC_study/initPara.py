import numpy as np
import os
import tempfile
import pandas as pd
import pyabc
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pyABC_study.ODE import ODESolver, euclidean_distance, normalise_data

# Read  and prepare raw data
ROOT_DIR = os.path.abspath(os.curdir)
rawData_path = ROOT_DIR + "/data/rawData.csv"
rawData = pd.read_csv(rawData_path)
rawData = rawData.astype("float32")
timePoints = rawData.iloc[:, 0].to_numpy()
rawDataDict = rawData.iloc[:, 1:].to_dict(orient='list')

expData = rawDataDict
for k in expData:
    expData[k] = np.array(expData[k])

# normalise_data(expData)

print("Target data")
print(expData)

# Run a rough inference on the raw data

solver = ODESolver()
solver.timePoint = timePoints
paraInit = {"lambdaN": 13.753031, "kNB": 1.581684, "muN": 2.155420, "vNM": 0.262360,
            "lambdaM": 2.993589, "kMB": 2041040, "muM": 2.201963,
            "sBN": 1.553020, "iBM": 2046259, "muB": 1.905163,
            "sAM": 11.001731, "muA": 22.022678}
testData = solver.ode_model(paraInit)

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

abc = pyabc.ABCSMC(models=solver.ode_model,
                   parameter_priors=paraPrior,
                   distance_function=euclidean_distance,
                   #distance_function=distance_adaptive,
                   population_size=500,
                   eps=pyabc.MedianEpsilon(100, median_multiplier=1)
                   )
db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))
abc.new(db_path, expData)

max_population = 17


history = abc.run(minimum_epsilon=0.1, max_nr_populations=max_population)

pyabc.visualization.plot_acceptance_rates_trajectory(history)
plt.show()

pyabc.visualization.plot_epsilons(history)
plt.show()

df, w = history.get_distribution(t=16)
df.mean()

"""
name
iBM        21.158021
kMB        33.334623
kNB        36.124380
lambdaM    34.842738
lambdaN    17.013443
muA        35.837526
muB         2.059064
muM        35.262282
muN         4.740578
sAM        27.802149
sBN        32.040063
vNM         4.524078
dtype: float64
"""