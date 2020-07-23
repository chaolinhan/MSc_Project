import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyabc
from pyabc.weighted_statistics import effective_sample_size

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
db_path = "sqlite:///db/model3_m_log.db"

history = pyabc.History(db_path)

print("ID: %d, generations: %d" % (history.id, history.max_t))

# %% Plot

solver = ODESolver()

# TODO change model name
solver.ode_model = solver.ode_model3

result_data(history, solver, nr_population=history.max_t, savefig=True)

# TODO change prior name
result_plot(history, None, para_prior3, history.max_t, savefig=True)


# %% Compare

history_1 = pyabc.History("sqlite:///db/model1_m_log.db")
history_2 = pyabc.History("sqlite:///db/model2_m_log.db")
history_3 = pyabc.History("sqlite:///db/model3_m_log.db")
history_list = [history_1, history_2, history_3]

history_label = ['model 1', 'model 2', 'model 3']

plt.style.use('default')
pyabc.visualization.plot_sample_numbers(history_list, labels=history_label, size=(4, 4))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
# plt.savefig("size1.png", dpi=200)
plt.show()

pyabc.visualization.plot_acceptance_rates_trajectory(history_list, labels=history_label, size=(6, 3))
plt.savefig("acc123.png", dpi=200)
plt.show()

pyabc.visualization.plot_epsilons(history_list, labels=history_label, size=(6, 3))
plt.savefig("eps123.png", dpi=200)
plt.show()

pyabc.visualization.plot_effective_sample_sizes(history_list, labels=history_label)
plt.show()

w = history_2.get_weighted_distances(t=history_2.max_t)['w']
ess = effective_sample_size(w)



# %% Some test

# df, w = history.get_distribution(t=history.max_t)
#
# pyabc.visualization.plot_kde_2d(df, w, x="mu_beta", y="s_beta_n")
# plt.show()
#
#
# from sklearn.linear_model import LinearRegression
#
# lr_fit = LinearRegression().fit(df["mu_beta"].to_numpy().reshape(-1,1), df["s_beta_n"].to_numpy().reshape(-1,1))
# Y = lr_fit.predict(df["mu_beta"].to_numpy().reshape(-1, 1))
#
#
# plt.scatter(df["mu_beta"], df["s_beta_n"])
# plt.scatter(df["mu_beta"], Y)
# plt.xlabel("μ_β")
# plt.ylabel("s_βN")
# plt.legend(["Samples", "y=0.741x+0.240"])
# plt.show()



# %% Model compare plot
pyabc.visualization.plot_epsilons(history)
plt.show()

pyabc.visualization.plot_acceptance_rates_trajectory(history)
plt.show()

pyabc.visualization.plot_model_probabilities(history)
plt.show()

