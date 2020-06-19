import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyabc

from pyABC_study.ODE import ODESolver, PriorLimits, arr2d_to_dict
from pyABC_study.dataPlot import result_data, result_plot

# %% Settings

lim = PriorLimits(0, 20)
lim2 = PriorLimits(0, 1)
lim3 = PriorLimits(0, 10)
# lim2 = PriorLimits(0, 20)
# lim3 = PriorLimits(0, 20)

para_prior = pyabc.Distribution(
    lambdaN=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    kNB=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    muN=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
    vNM=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
    lambdaM=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    kMB=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
    muM=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
    sBN=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    iBM=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    muB=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    sAM=pyabc.RV("uniform", lim.lb, lim.interval_length),
    muA=pyabc.RV("uniform", lim.lb, lim.interval_length)
)

lim3 = PriorLimits(0, 20)
lim2 = PriorLimits(0, 20)


para_prior_wide = pyabc.Distribution(
    lambdaN=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    kNB=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    muN=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
    vNM=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
    lambdaM=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    kMB=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
    muM=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
    sBN=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    iBM=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    muB=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    sAM=pyabc.RV("uniform", lim.lb, lim.interval_length),
    muA=pyabc.RV("uniform", lim.lb, lim.interval_length)
)

solver = ODESolver()

obs_data_noisy_s = solver.ode_model(para_true, flatten=False, add_noise=True)
obs_data_raw_s = solver.ode_model(para_true, flatten=False, add_noise=False)

solver.time_point = solver.time_point_exp
obs_data_raw_s_less = solver.ode_model(para_true, flatten=False, add_noise=False)
#
# print("Target data")
# print(obs_data_noisy_s)

# %% Load database

db_path = "sqlite:///db/abcsmc_test.db"

history = pyabc.History(db_path)

print("ID: %d, generations: %d" % (history.id, history.max_t))

# %% Plot

raw_data_path = os.path.abspath(os.curdir) + "/data/rawData.csv"
raw_data = pd.read_csv(raw_data_path).astype("float32")

time_points: object = raw_data.iloc[:, 0].to_numpy()
exp_data = raw_data.iloc[:, 1:].to_numpy()
exp_data = arr2d_to_dict(exp_data)

exp_data_s = raw_data.iloc[:, 1:].to_dict(orient='list')
for k in exp_data_s:
    exp_data_s[k] = np.array(exp_data_s[k])

result_data(history, exp_data_s, solver.time_point_exp, history.max_t)

lim = PriorLimits(0, 75)

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

result_plot(history, para_true, paraPrior, history.max_t)
#
# %% kernel compare
#
# db_path_MNN = os.listdir('db/')
#
# history_base = pyabc.History('sqlite:///db/MNN_base.db')
# history_base_scale = pyabc.History('sqlite:///db/MNN_base_scale.db')
# history_500 = pyabc.History('sqlite:///db/MNN_500.db')
# history_100 = pyabc.History('sqlite:///db/MNN_100.db')
# history_50 = pyabc.History('sqlite:///db/MNN_50.db')
# history_250 = pyabc.History('sqlite:///db/MNN_250.db')
# history_750 = pyabc.History('sqlite:///db/MNN_750.db')
#
# history_list = [history_base, history_base_scale, history_750, history_500, history_250, history_100, history_50]
# history_label = ['Multivariate Normal', 'Multivariate Normal\nscale=0.5', 'NN M=750', 'NN M=500', 'NN M=250', 'NN M=100',
#                  'NN M=50']
#
# # %% Plot
# pyabc.visualization.plot_sample_numbers(history_list, labels=history_label, size=(12, 6))
# plt.show()
#
# pyabc.visualization.plot_effective_sample_sizes(history_list, labels=history_label)
# plt.show()
#
# pyabc.visualization.plot_acceptance_rates_trajectory(history_list, labels=history_label)
# plt.show()
#
# pyabc.visualization.plot_epsilons(history_list, labels=history_label)
# plt.show()
#
# pyabc.visualization.plot_total_sample_numbers(history_list, labels=history_label)
# plt.show()
#
# for item in history_list:
#     # result_plot(item, para_true, paraPrior, item.max_t)
#     result_data(item, obs_data_raw_s, solver.timePoint, item.max_t)

# %% kernel median eps compare

# history_base = pyabc.History('sqlite:///db/MNN_base_median.db')
# history_base_scale = pyabc.History('sqlite:///db/MNN_base_scale_median.db')
# history_base_GS = pyabc.History('sqlite:///db/MNN_base_GS_median.db')
# history_500 = pyabc.History('sqlite:///db/MNN_500_median.db')
# history_100 = pyabc.History('sqlite:///db/MNN_100_median.db')
# history_50 = pyabc.History('sqlite:///db/MNN_50_median.db')
# history_250 = pyabc.History('sqlite:///db/MNN_250_median.db')
# history_750 = pyabc.History('sqlite:///db/MNN_750_median.db')
#
# history_list = [history_base, history_base_scale, history_base_GS, history_750, history_500, history_250, history_100,
#                 history_50]
# history_label = ['Multivariate Normal', 'Multivariate Normal\nscale=0.5', 'Multivariate Normal\nGridSearch', 'NN M=750',
#                  'NN M=500', 'NN M=250', 'NN M=100',
#                  'NN M=50']

# %% Plot

# pyabc.visualization.plot_sample_numbers(history_list, labels=history_label, size=(16, 6))
# plt.show(scale=2)
#
# pyabc.visualization.plot_effective_sample_sizes(history_list, labels=history_label)
# plt.show()
#
# pyabc.visualization.plot_acceptance_rates_trajectory(history_list, labels=history_label)
# plt.show()
#
# pyabc.visualization.plot_epsilons(history_list, labels=history_label)
# plt.show()
#
# pyabc.visualization.plot_total_sample_numbers(history_list, labels=history_label)
# plt.show()
#
# for item in history_list:
#     # result_plot(item, para_true, paraPrior, item.max_t)
#     result_data(item, obs_data_raw_s, solver.timePoint, item.max_t)

# %% Adaptive distance compare
#
# history_base = pyabc.History('sqlite:///db/SMC_base.db', )
# history_f = pyabc.History('sqlite:///db/SMC_f.db')
# history_f2 = pyabc.History('sqlite:///db/SMC_f2.db')
# # history_a = pyabc.History('sqlite:///db/SMC_a.db')
# # history_af = pyabc.History('sqlite:///db/SMC_af.db')
# #
# history_list = [history_base, history_f, history_f2]
#
# history_label = ['No factor', '+ Range factor', '+ Variance factor']

# %% Plot

# pyabc.visualization.plot_sample_numbers(history_list, labels=history_label, size=(12, 6))
# plt.show(scale=2)
#
# pyabc.visualization.plot_effective_sample_sizes(history_list, labels=history_label)
# plt.show()
#
# pyabc.visualization.plot_acceptance_rates_trajectory(history_list, labels=history_label)
# plt.show()
#
# pyabc.visualization.plot_epsilons(history_list, labels=history_label)
# plt.show()
#
# pyabc.visualization.plot_total_sample_numbers(history_list, labels=history_label)
# plt.show()
#
# for item in history_list:
#     # result_plot(item, para_true, paraPrior, item.max_t)
#     result_data(item, obs_data_raw_s, solver.timePoint, item.max_t)

# %% Prior range and data size compare

history_base = pyabc.History('sqlite:///db/SMC_base_big.db', )
history_less = pyabc.History('sqlite:///db/SMC_base_big_less.db')
history_wide = pyabc.History('sqlite:///db/SMC_base_big_wide.db')
history_list = [history_base, history_less, history_wide]

history_label = ['base', 'less data', 'wider prior\nrange']

# %% Plot

pyabc.visualization.plot_sample_numbers(history_list, labels=history_label, size=(12, 6))
plt.show(scale=2)

pyabc.visualization.plot_effective_sample_sizes(history_list, labels=history_label)
plt.show()

pyabc.visualization.plot_acceptance_rates_trajectory(history_list, labels=history_label)
plt.show()

pyabc.visualization.plot_epsilons(history_list, labels=history_label)
plt.show()

pyabc.visualization.plot_total_sample_numbers(history_list, labels=history_label)
plt.show()

result_data(history_base, obs_data_raw_s, solver.time_point_default, history_base.max_t)
result_data(history_less, obs_data_raw_s_less, solver.time_point_exp, history_base.max_t)
result_data(history_wide, obs_data_raw_s, solver.time_point_default, history_base.max_t)

result_plot(history_base, para_true, para_prior, history_base.max_t)
result_plot(history_less, para_true, para_prior, history_less.max_t)
result_plot(history_wide, para_true, para_prior_wide, history_wide.max_t)