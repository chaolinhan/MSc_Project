import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabc
from scipy import optimize

from pyABC_study.ODE import ODESolver, euclidean_distance, PriorLimits


def arr2d_to_dict(arr: np.ndarray):
    return {i: arr.flatten()[i] for i in range(arr.flatten().__len__())}


def para_list_to_dict(para):
    para_dict = {
        'iBM': para[0],
        'kMB': para[1],
        'kNB': para[2],
        'lambdaM': para[3],
        'lambdaN': para[4],
        'muA': para[5],
        'muB': para[6],
        'muM': para[7],
        'muN': para[8],
        'sAM': para[9],
        'sBN': para[10],
        'vNM': para[11]
    }
    return para_dict


# Read  and prepare raw data

rawData_path = os.path.abspath(os.curdir) + "/data/rawData.csv"
rawData = pd.read_csv(rawData_path).astype("float32")

timePoints: object = rawData.iloc[:, 0].to_numpy()
expData = rawData.iloc[:, 1:].to_numpy()
expData = arr2d_to_dict(expData)

expData_no_f = rawData.iloc[:, 1:].to_dict(orient='list')
for k in expData_no_f:
    expData_no_f[k] = np.array(expData_no_f[k])

# normalise_data(expData)
print("Target data")
print(expData)


# Run a rough inference on the raw data

solver = ODESolver()
# Reload the timepoints to be calculated in ODE solver
solver.time_point = timePoints
#
# lim = PriorLimits(0, 100)
#
# paraPrior = pyabc.Distribution(
#     lambdaN=pyabc.RV("uniform", lim.lb, lim.interval_length),
#     kNB=pyabc.RV("uniform", lim.lb, lim.interval_length),
#     muN=pyabc.RV("uniform", lim.lb, lim.interval_length),
#     vNM=pyabc.RV("uniform", lim.lb, lim.interval_length),
#     lambdaM=pyabc.RV("uniform", lim.lb, lim.interval_length),
#     kMB=pyabc.RV("uniform", lim.lb, lim.interval_length),
#     muM=pyabc.RV("uniform", lim.lb, lim.interval_length),
#     sBN=pyabc.RV("uniform", lim.lb, lim.interval_length),
#     iBM=pyabc.RV("uniform", lim.lb, lim.interval_length),
#     muB=pyabc.RV("uniform", lim.lb, lim.interval_length),
#     sAM=pyabc.RV("uniform", lim.lb, lim.interval_length),
#     muA=pyabc.RV("uniform", lim.lb, lim.interval_length)
# )
#
# #
#
# distanceP2_adaptive = pyabc.AdaptivePNormDistance(p=2,
#                                                   scale_function=pyabc.distance.root_mean_square_deviation
#                                                   )
# distanceP2 = pyabc.PNormDistance(p=2)
#
# abc = pyabc.ABCSMC(models=solver.ode_model,
#                    parameter_priors=paraPrior,
#                    distance_function=distanceP2,
#                    population_size=1200,
#                    eps=pyabc.MedianEpsilon(100, median_multiplier=1)
#                    )
# #
# db_path = ("sqlite:///" +
#            os.path.join(tempfile.gettempdir(), "test.db"))
# abc.new(db_path, expData)
#
# max_population = 15
# #
# history = abc.run(minimum_epsilon=0.1, max_nr_populations=max_population)
# #
# # Plot the results
#
# pyabc.visualization.plot_acceptance_rates_trajectory(history)
# plt.show()
#
# pyabc.visualization.plot_epsilons(history)
# plt.show()
#
# df, w = history.get_distribution(t=max_population - 1)
#
# np.argmax()
#
# pyabc.visualization.plot_kde_matrix(df, w)
# plt.show()
#
# # Print results
#
# # Mean of last population
# print(df.mean())
#
# # Sum of weight * particles
# for i in range(12):
#     print(df.iloc[:, i].name + '\t\t%.6f' % (df.iloc[:, i] * w).sum())
#
# # Particle with maximal weight
# print(df.iloc[w.argmax(), :])

# """
# Output from one run:
#
#
# mean() method:
#
# iBM         9.737715
# kMB        47.458632
# kNB         9.973562
# lambdaM    39.107247
# lambdaN    25.262527
# muA        47.041885
# muB        15.762823
# muM        39.539603
# muN        79.994190
# sAM        38.563204
# sBN        37.618288
# vNM        13.229111
#
#
# df*w sum() method:
#
# iBM		9.051270
# kMB		40.881926
# kNB		9.618762
# lambdaM		41.405661
# lambdaN		29.360990
# muA		44.426018
# muB		16.450285
# muM		37.356256
# muN		78.150011
# sAM		33.580249
# sBN		41.486109
# vNM		13.005909
#
# maximal weight method:
# iBM         6.706790
# kMB        37.790301
# kNB        13.288773
# lambdaM    40.238402
# lambdaN    45.633238
# muA        39.136272
# muB        15.821665
# muM        34.883162
# muN        77.583389
# sAM        40.198178
# sBN        32.110228
# vNM        12.689222
# """

# Least squares

#
# def residual(x, iBM, kMB, kNB, lambdaM, lambdaN, muA, muB, muM, muN, sAM, sBN, vNM):
#     paraDict = {
#         'iBM': iBM,
#         'kMB': kMB,
#         'kNB': kNB,
#         'lambdaM': lambdaM,
#         'lambdaN': lambdaN,
#         'muA': muA,
#         'muB': muB,
#         'muM': muM,
#         'muN': muN,
#         'sAM': sAM,
#         'sBN': sBN,
#         'vNM': vNM
#     }
#     simulationData = solver.ode_model(paraDict)
#     # print(x)
#     # print(simulationData)
#     ans = np.array([])
#     for ii in range(12):
#         sim = {"N": simulationData["N"][ii],
#                "M": simulationData["M"][ii],
#                "B": simulationData["B"][ii],
#                "A": simulationData["A"][ii]}
#         ydata = {"N": expData["N"][ii],
#                  "M": expData["M"][ii],
#                  "B": expData["B"][ii],
#                  "A": expData["A"][ii]}
#         # print(euclidean_distance(ydata, sim))
#         ans = np.append(ans, euclidean_distance(ydata, sim))
#     return ans
#
#
# xdata = np.array(range(12))
# residual(xdata, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10)
# ydata = np.zeros(12)
#
# paraGuess = [10] * 12
#
# popt, pcov = optimize.curve_fit(residual, xdata, ydata, p0=paraGuess, bounds=(0, 100))
#
# plt.plot(xdata, residual(xdata, *popt), 'r-')
# plt.show()

# paraGuess = [5, 37, 13, 40, 45, 40, 15, 35, 77, 40, 32, 12]
paraGuess = [25] * 12

distanceP2 = pyabc.PNormDistance(p=2)
tmpData = solver.ode_model(para_list_to_dict(paraGuess))


def residual_ls(para: list):
    para_dict = para_list_to_dict(para)
    simulation_data = solver.ode_model(para_dict)
    # print(x)
    # print(simulation_data)
    # return distanceP2(simulation_data, expData)
    ans = [abs(np.nan_to_num((simulation_data[i] - expData[i]))) for i in range(48)]
    return ans
# TODO BUG spotted: para changed but ans not


#%% Run LS fitting

paraGuess = np.array([5] * 12)

paraEST = optimize.least_squares(residual_ls, paraGuess, method='trf', bounds=(1e-3, 20))

print(paraEST.x)

paraDict = para_list_to_dict(paraEST.x)
solver.time_point = solver.time_point_default
simulationData = solver.ode_model(paraDict, flatten=False)

plt.plot(solver.time_point, simulationData['N'], solver.time_point, simulationData['M'])
plt.scatter(rawData['time'][:-1], rawData['N'][:-1], label="N")
plt.scatter(rawData['time'][:-1], rawData['M'][:-1], label="Phi")
plt.legend()
plt.show()

plt.plot(solver.time_point, simulationData['B'], solver.time_point, simulationData['A'])
plt.scatter(rawData['time'], rawData['B'], label="beta")
plt.scatter(rawData['time'], rawData['A'], label="alpha")
plt.legend()
plt.show()
