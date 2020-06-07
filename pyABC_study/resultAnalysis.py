import os

import pyabc

from pyABC_study.ODE import ODESolver, PriorLimits
from pyABC_study.dataPlot import obs_data_plot, result_plot, result_data

#%% Settings

para_true = {'iBM': 1.0267462374320455,
             'kMB': 0.07345932286118964,
             'kNB': 2.359199465995228,
             'lambdaM': 2.213837884117815,
             'lambdaN': 7.260925726829641,
             'muA': 18.94626522780349,
             'muB': 2.092860392215201,
             'muM': 0.17722330053184654,
             'muN': 0.0023917569160019844,
             'sAM': 10.228522400429998,
             'sBN': 4.034313992927392,
             'vNM': 0.3091883041193706}

lim = PriorLimits(0, 20)
lim2 = PriorLimits(0, 1)
lim3 = PriorLimits(0, 10)
# lim2 = PriorLimits(0, 20)
# lim3 = PriorLimits(0, 20)

paraPrior = pyabc.Distribution(
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
# obs_data_noisy = solver.ode_model(para_true, flatten=True, add_noise=True)
# obs_data_raw = solver.ode_model(para_true, flatten=True, add_noise=False)

obs_data_noisy_s = solver.ode_model(para_true, flatten=False, add_noise=True)
# obs_data_raw_s = solver.ode_model(para_true, flatten=False, add_noise=False)
#
# print("Target data")
# print(obs_data_noisy_s)

#%% Load database

db_path = "sqlite:///base.db"

history = pyabc.History(db_path)

print("ID: %d, generations: %d" %(history.id, history.max_t))

#%% Plot

result_data(history, obs_data_noisy_s, solver.timePoint, history.max_t)
result_plot(history, para_true, paraPrior, history.max_t)