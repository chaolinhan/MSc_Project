import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabc

from pyABC_study.ODE import ODESolver, PriorLimits, arr2d_to_dict, para_true1, para_prior
from pyABC_study.dataPlot import result_data_old, result_plot, result_data

# %% Settings

# lim = PriorLimits(0, 20)
# lim2 = PriorLimits(0, 1)
# lim3 = PriorLimits(0, 10)
# lim2 = PriorLimits(0, 20)
# lim3 = PriorLimits(0, 20)

# para_prior = pyabc.Distribution(
#     lambdaN=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
#     kNB=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
#     muN=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
#     vNM=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
#     lambdaM=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
#     kMB=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
#     muM=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
#     sBN=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
#     iBM=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
#     muB=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
#     sAM=pyabc.RV("uniform", lim.lb, lim.interval_length),
#     muA=pyabc.RV("uniform", lim.lb, lim.interval_length)
# )

lim = PriorLimits(1e-6, 50)

prior_distribution = "uniform"

print(prior_distribution)

para_prior1 = para_prior(lim, prior_distribution, 1)

#
# lim3 = PriorLimits(0, 20)
# lim2 = PriorLimits(0, 20)
#
# para_prior_wide = pyabc.Distribution(
#     lambdaN=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
#     kNB=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
#     muN=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
#     vNM=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
#     lambdaM=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
#     kMB=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
#     muM=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
#     sBN=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
#     iBM=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
#     muB=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
#     sAM=pyabc.RV("uniform", lim.lb, lim.interval_length),
#     muA=pyabc.RV("uniform", lim.lb, lim.interval_length)
# )

solver = ODESolver()
solver.time_point = solver.time_point_default

obs_data_raw_s = solver.ode_model1(para_true1, flatten=False, add_noise=False)

solver.time_point = solver.time_point_exp
obs_data_raw_s_less = solver.ode_model(para_true1, flatten=False, add_noise=False)
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
db_path_MNN = os.listdir('db/infer_back/')

history_base = pyabc.History('sqlite:///db/infer_back/MNN_base.db')
history_base_scale = pyabc.History('sqlite:///db/infer_back/MNN_base_scale.db')
history_500 = pyabc.History('sqlite:///db/infer_back/MNN_500.db')
history_100 = pyabc.History('sqlite:///db/infer_back/MNN_100.db')
history_50 = pyabc.History('sqlite:///db/infer_back/MNN_50.db')
history_250 = pyabc.History('sqlite:///db/infer_back/MNN_250.db')
history_750 = pyabc.History('sqlite:///db/infer_back/MNN_750.db')

history_list = [history_base, history_base_scale, history_750, history_500, history_250, history_100, history_50]
history_label = ['Multivariate Normal', 'Multivariate Normal\nscale=0.5', 'Local NN M=750', 'Local NN M=500',
                 'Local NN M=250', 'Local NN M=100',
                 'Local NN M=50']

# %% Plot
plt.style.use('default')
pyabc.visualization.plot_sample_numbers(history_list, labels=history_label, size=(15, 6))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
plt.savefig("kernel2.png", dpi=200)
plt.show()

pyabc.visualization.plot_effective_sample_sizes(history_list, labels=history_label)
plt.show()

pyabc.visualization.plot_acceptance_rates_trajectory(history_list, labels=history_label, size=(12, 6))
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
plt.xticks(range(0, 25, 2))
plt.savefig("acceptance2.png", dpi=200)
plt.show()

pyabc.visualization.plot_epsilons(history_list, labels=history_label)
plt.show()

pyabc.visualization.plot_total_sample_numbers(history_list, labels=history_label)
plt.show()

for item in history_list:
    # result_plot(item, para_true, paraPrior, item.max_t)
    result_data(item, obs_data_raw_s, solver.timePoint, item.max_t)

# %% kernel median eps compare

history_base = pyabc.History('sqlite:///db/infer_back/MNN_base_median.db')
history_base_scale = pyabc.History('sqlite:///db/infer_back/MNN_base_scale_median.db')
history_base_GS = pyabc.History('sqlite:///db/infer_back/MNN_base_GS_median.db')
history_500 = pyabc.History('sqlite:///db/infer_back/MNN_500_median.db')
history_100 = pyabc.History('sqlite:///db/infer_back/MNN_100_median.db')
history_50 = pyabc.History('sqlite:///db/infer_back/MNN_50_median.db')
history_250 = pyabc.History('sqlite:///db/infer_back/MNN_250_median.db')
history_750 = pyabc.History('sqlite:///db/infer_back/MNN_750_median.db')

history_list = [history_base, history_base_scale, history_base_GS, history_750, history_500, history_250, history_100,
                history_50]
history_label = ['Multivariate Normal', 'Multivariate Normal\nscale=0.5', 'Multivariate Normal\nGridSearch', 'NN M=750',
                 'NN M=500', 'NN M=250', 'NN M=100',
                 'NN M=50']

# %% Plot

pyabc.visualization.plot_sample_numbers(history_list, labels=history_label, size=(16, 6))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
plt.savefig("kernel2.png", dpi=200)
plt.show(scale=2)

pyabc.visualization.plot_effective_sample_sizes(history_list, labels=history_label)
plt.show()

pyabc.visualization.plot_acceptance_rates_trajectory(history_list, labels=history_label)
plt.show()

pyabc.visualization.plot_epsilons(history_list, labels=history_label)
plt.show()

pyabc.visualization.plot_total_sample_numbers(history_list, labels=history_label)
plt.show()

for item in history_list:
    # result_plot(item, para_true, paraPrior, item.max_t)
    result_data(item, obs_data_raw_s, solver.timePoint, item.max_t)

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

history_base = pyabc.History('sqlite:///db/infer_back/SMC_base_big.db', )
history_less = pyabc.History('sqlite:///db/infer_back/SMC_base_big_less.db')
history_wide = pyabc.History('sqlite:///db/infer_back/SMC_base_big_wide.db')
history_list = [history_base, history_less, history_wide]

history_label = ['standard', 'less data', 'wider prior\nrange']

# %% Plot

plt.style.use('default')
pyabc.visualization.plot_sample_numbers(history_list, labels=history_label, size=(4, 4))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
plt.savefig("size1.png", dpi=200)
plt.show()

pyabc.visualization.plot_effective_sample_sizes(history_list, labels=history_label)
plt.show()

pyabc.visualization.plot_acceptance_rates_trajectory(history_list, labels=history_label)
plt.show()

pyabc.visualization.plot_epsilons(history_list, labels=history_label, size=(4, 4))
plt.savefig("size2.png", dpi=200)
plt.show()

pyabc.visualization.plot_total_sample_numbers(history_list, labels=history_label)
plt.show()


# %% Plot curve
history_base = pyabc.History('sqlite:///db/infer_back/ib_base.db')

solver.time_point = solver.time_point_default
result_data_old(history_base, solver, obs_data_raw_s, history_base.max_t)

pyabc.visualization.plot_epsilons(history_base)
# plt.savefig("size2.png", dpi=200)
plt.show()

# solver.time_point = solver.time_point_exp
# result_data_old(history_less, solver, obs_data_raw_s_less, history_base.max_t)
#
# solver.time_point = solver.time_point_default
# result_data_old(history_wide, solver, obs_data_raw_s, history_base.max_t)


# %% Plot parameters
result_plot(history_base, para_true1, para_prior, history_base.max_t)
result_plot(history_less, para_true1, para_prior, history_less.max_t)
result_plot(history_wide, para_true1, para_prior_wide, history_wide.max_t)
