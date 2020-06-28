import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyabc

from pyABC_study.ODE import ODESolver, PriorLimits, arr2d_to_dict, exp_data, exp_data_s, para_prior
from pyABC_study.dataPlot import result_data, result_plot

# %% Settings

# TODO: change prior
lim = PriorLimits(1e-6, 50)
prior_distribution = "loguniform"

print(prior_distribution)

para_prior1 = para_prior(lim, prior_distribution, 1)
para_prior2 = para_prior(lim, prior_distribution, 2)
para_prior3 = para_prior(lim, prior_distribution, 3)
para_prior4 = para_prior(lim, prior_distribution, 4)
para_prior5 = para_prior(lim, prior_distribution, 5)



# %% Load database

# TODO change database name
db_path = "sqlite:///db/model5_m_log_lp.db"

history = pyabc.History(db_path)

print("ID: %d, generations: %d" % (history.id, history.max_t))

# %% Plot

solver = ODESolver()

# TODO change model name
solver.ode_model = solver.ode_model5

result_data(history, solver, nr_population=history.max_t)

# TODO change prior name
result_plot(history, None, para_prior5, history.max_t)


# %% Model compare plot
pyabc.visualization.plot_epsilons(history)
plt.show()

pyabc.visualization.plot_acceptance_rates_trajectory(history)
plt.show()

pyabc.visualization.plot_model_probabilities(history)
plt.show()