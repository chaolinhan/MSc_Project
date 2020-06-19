import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyabc

from pyABC_study.ODE import ODESolver, PriorLimits, arr2d_to_dict, exp_data, exp_data_s
from pyABC_study.dataPlot import result_data, result_plot

# %% Settings

lim = PriorLimits(1e-5, 75)

prior_distribution = "uniform"

print(prior_distribution)

para_prior1 = pyabc.Distribution(
    lambda_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    k_n_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    v_n_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    lambda_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    k_phi_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    s_beta_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    i_beta_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    s_alpha_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_alpha=pyabc.RV(prior_distribution, lim.lb, lim.interval_length)
)

# %% Load database

db_path = "sqlite:///db/model1.db"

history = pyabc.History(db_path)

print("ID: %d, generations: %d" % (history.id, history.max_t))

# %% Plot

solver = ODESolver()

solver.ode_model = solver.ode_model1

result_data(history, exp_data_s, solver, history.max_t-1)


result_plot(history, None, para_prior1, history.max_t-1)