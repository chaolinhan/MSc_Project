import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyabc

from pyABC_study.ODE import ODESolver, PriorLimits, arr2d_to_dict, exp_data, exp_data_s
from pyABC_study.dataPlot import result_data, result_plot

# %% Settings

# TODO: change prior
lim = PriorLimits(1e-6, 50)
prior_distribution = "loguniform"

print(prior_distribution)

# args = (para["lambda_n"], para["k_n_beta"], para["mu_n"], para["v_n_phi"],
#         para["lambda_phi"], para["k_phi_beta"], para["mu_phi"],
#         para["s_beta_n"], para["i_beta_phi"], para["mu_beta"],
#         para["s_alpha_phi"], para["mu_alpha"])

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

para_prior2 = pyabc.Distribution(
    lambda_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    a=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    k_n_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    v_n_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    k_phi_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    s_beta_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    i_beta_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    s_alpha_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_alpha=pyabc.RV(prior_distribution, lim.lb, lim.interval_length)
)

para_prior3 = pyabc.Distribution(
    lambda_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    a=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    k_n_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    v_n_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    k_phi_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    s_beta_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    s_alpha_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_alpha=pyabc.RV(prior_distribution, lim.lb, lim.interval_length)
)

# %% Load database

# TODO change database name
db_path = "sqlite:///db/model3_m_log.db"

history = pyabc.History(db_path)

print("ID: %d, generations: %d" % (history.id, history.max_t))

# %% Plot

solver = ODESolver()

# TODO change model name
solver.ode_model = solver.ode_model3

result_data(history, solver, nr_population=history.max_t)

# TODO change prior name
result_plot(history, None, para_prior3, history.max_t)


# %% Model compare plot
pyabc.visualization.plot_epsilons(history)
plt.show()

pyabc.visualization.plot_acceptance_rates_trajectory(history)
plt.show()

pyabc.visualization.plot_model_probabilities(history)
plt.show()