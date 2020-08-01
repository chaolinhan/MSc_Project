import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabc
from scipy import optimize

from pyABC_study.ODE import ODESolver, exp_data, exp_data_s, exp_data_SEM


def arr2d_to_dict(arr: np.ndarray):
    return {i: arr.flatten()[i] for i in range(arr.flatten().__len__())}


def para_list_to_dict(para):
    para_dict = {
        'i_beta_phi': para[0],
        'k_phi_beta': para[1],
        'k_n_beta': para[2],
        'lambda_phi': para[3],
        'lambda_n': para[4],
        'mu_alpha': para[5],
        'mu_beta': para[6],
        'mu_phi': para[7],
        'mu_n': para[8],
        's_alpha_phi': para[9],
        's_beta_n': para[10],
        'v_n_phi': para[11]
    }
    return para_dict


# Read  and prepare raw data
rawData_path = os.path.abspath(os.curdir) + "/data/rawData.csv"
rawData = pd.read_csv(rawData_path).astype("float32")
# normalise_data(expData)
print("Target data")
print(exp_data)

# Run a rough inference on the raw data

solver = ODESolver()
# Reload the timepoints to be calculated in ODE solver
solver.time_point = solver.time_point_exp
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
paraGuess = [1] * 12

distanceP2 = pyabc.PNormDistance(p=2)
tmpData = solver.ode_model(para_list_to_dict(paraGuess))


def residual_ls(para: list):
    para_dict = para_list_to_dict(para)
    simulation_data = solver.ode_model(para_dict)
    # print(x)
    # print(simulation_data)
    # return distanceP2(simulation_data, expData)
    ans = [abs(np.nan_to_num((simulation_data[i] - exp_data[i]))) for i in range(48)]
    return ans


# TODO BUG spotted: para changed but ans not


# %% Run LS fitting

paraGuess = np.array([5] * 12)

paraEST = optimize.least_squares(residual_ls, paraGuess, method='trf', bounds=(1e-3, 20))

print(paraEST.x)

paraDict = para_list_to_dict(paraEST.x)
solver.time_point = solver.time_point_default
simulationData = solver.ode_model(paraDict, flatten=False)

# %% Plot
plt.style.use('default')
plt.plot(solver.time_point, simulationData['N'], label="$N$ simulated")
plt.plot(solver.time_point, simulationData['M'], label="$Φ$ simulated")
plt.scatter(rawData['time'][:-1], rawData['N'][:-1], color='b', linewidths=1.2, label="$N$ observed")
plt.errorbar(solver.time_point_exp, exp_data_s['N'],
             yerr=[[0.5 * x for x in exp_data_SEM['N']],
                   [0.5 * x for x in exp_data_SEM['N']]], fmt='none',
             ecolor='b', elinewidth=2, alpha=0.4)
plt.errorbar(solver.time_point_exp, exp_data_s['M'],
             yerr=[[0.5 * x for x in exp_data_SEM['M']],
                   [0.5 * x for x in exp_data_SEM['M']]], fmt='none',
             ecolor='r', elinewidth=2, alpha=0.4)
plt.scatter(rawData['time'][:-1], rawData['M'][:-1], color='r', linewidths=1.2, label="$Φ$ observed")
plt.xlim(-5, 120)
plt.ylim(0, None)
plt.grid(alpha=0.2)
plt.legend()
plt.savefig("LS1.png", dpi=200)
plt.show()

plt.style.use('default')
plt.plot(solver.time_point, simulationData['B'], label="$β$ simulated")
plt.plot(solver.time_point, simulationData['A'], label="$α$ simulated")
plt.scatter(rawData['time'], rawData['B'], color='b', linewidths=1.2, label="$β$ observed")
plt.scatter(rawData['time'], rawData['A'], color='r', linewidths=1.2, label="$α$ observed")
plt.errorbar(solver.time_point_exp, exp_data_s['B'],
             yerr=[[0.5 * x for x in exp_data_SEM['B']],
                   [0.5 * x for x in exp_data_SEM['B']]], fmt='none',
             ecolor='b', elinewidth=2, alpha=0.4)
plt.errorbar(solver.time_point_exp, exp_data_s['A'],
             yerr=[[0.5 * x for x in exp_data_SEM['A']],
                   [0.5 * x for x in exp_data_SEM['A']]], fmt='none',
             ecolor='r', elinewidth=2, alpha=0.4)
plt.xlim(-5, 120)
plt.ylim(0, None)
plt.grid(alpha=0.2)
plt.legend()
plt.savefig("LS2.png", dpi=200)
plt.show()

### True
# {'i_beta_phi': 1.7062457206228092,
#  'k_phi_beta': 0.12351843450795977,
#  'k_n_beta': 3.962697253452481,
#  'lambda_phi': 1.314620449006551,
#  'lambda_n': 2.198916934929738,
#  'mu_alpha': 19.6642270425759,
#  'mu_beta': 0.5212435818886882,
#  'mu_phi': 0.14542950112701297,
#  'mu_n': 1.7219112772837462,
#  's_alpha_phi': 10.241631079619625,
#  's_beta_n': 6.553614835929477,
#  'v_n_phi': 0.21949169250522468}

# {'N': array([ 0.        ,  1.45214314,  3.34310534,  9.54284681, 26.010689  ,
#         25.32439927, 13.22325301,  7.04014959,  4.07797329,  2.6220302 ,
#          1.89712109,  1.54331922,  1.38587988,  1.33597784,  1.34599707,
#          1.38875301,  1.51218252,  1.6329903 ,  1.72349665,  1.77980132,
#          1.80916514,  1.82144842,  1.82475136,  1.82426837,  1.82274257,
#          1.82133795,  1.82038237,  1.8198467 ,  1.81955671,  1.81957274]),
#  'M': array([ 0.        ,  0.36023119,  0.74440936,  1.66674259,  4.35285748,
#         10.76896793, 14.40369887, 15.37411485, 15.03594774, 14.22826163,
#         13.34562508, 12.54737821, 11.88372282, 11.35806593, 10.95574607,
#         10.65671868, 10.29015876, 10.12477172, 10.06751198, 10.0603397 ,
#         10.07100379, 10.08419743, 10.0943439 , 10.1005714 , 10.10375071,
#         10.10504624, 10.10537365, 10.10530254, 10.10477313, 10.10477997]),
#  'B': array([ 1.        ,  1.68823936,  3.32160438,  8.3855355 , 20.29084442,
#         23.73400683, 14.63576954,  7.97346631,  4.35773555,  2.52381331,
#          1.60879341,  1.15856511,  0.94776705,  0.8643185 ,  0.85052996,
#          0.874978  ,  0.97224809,  1.07727057,  1.1596786 ,  1.21284293,
#          1.24164902,  1.25437699,  1.2583099 ,  1.25832817,  1.25711561,
#          1.25587357,  1.25498595,  1.25446879,  1.25415112,  1.25416696]),
#  'A': array([1.        , 0.15668876, 0.34586147, 0.81377964, 2.18369889,
#         5.53662493, 7.47367391, 8.00511368, 7.83979533, 7.42214987,
#         6.96209554, 6.54476227, 6.19724758, 5.92171514, 5.71066527,
#         5.55368556, 5.36103029, 5.27390405, 5.24358671, 5.23964698,
#         5.24514669, 5.25202702, 5.25733772, 5.26060516, 5.26227734,
#         5.26296122, 5.26313596, 5.26310036, 5.26282364, 5.26282718])}
