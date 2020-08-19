import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pyabc
from pyabc.weighted_statistics import effective_sample_size

from pyABC_study.ODE import ODESolver, PriorLimits, para_prior
from pyABC_study.dataPlot import result_data, result_plot, result_plot_sp

# %% Settings

# TODO: change prior
lim = PriorLimits(1e-6, 25)
prior_distribution = "loguniform"

print(prior_distribution)

para_prior1 = para_prior(lim, prior_distribution, 1)
para_prior2 = para_prior(lim, prior_distribution, 2)
para_prior3 = para_prior(lim, prior_distribution, 3)
para_prior4 = para_prior(lim, prior_distribution, 4)
para_prior5 = para_prior(lim, prior_distribution, 5)

# %% Load database

# TODO change database name
db_path = "sqlite:///db/model5_24_more.db"

history = pyabc.History(db_path)

print("ID: %d, generations: %d" % (history.id, history.max_t))

# %% Plot

solver = ODESolver()

# TODO change model name
solver.ode_model = solver.ode_model5

result_data(history, solver, nr_population=history.max_t, savefig=True)

# TODO change prior name
result_plot(history, None, para_prior5, history.max_t - 5, savefig=False)

df, w = history.get_distribution(t=history.max_t - 5)
pyabc.visualization.plot_kde_matrix(df, w)
plt.show()

pyabc.visualization.plot_epsilons(history)
plt.show()

# %% Compare

history_1 = pyabc.History("sqlite:///db/model3_m_log.db")
history_2 = pyabc.History("sqlite:///db/model4_m_log.db")
history_3 = pyabc.History("sqlite:///db/model5_m_log.db")
history_list = [history_1, history_2, history_3]

history_label = ['model 3', 'model 4', 'model 5']

plt.style.use('default')
pyabc.visualization.plot_sample_numbers(history_list, labels=history_label, size=(4, 4))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
# plt.savefig("size1.png", dpi=200)
plt.show()

pyabc.visualization.plot_acceptance_rates_trajectory(history_list, labels=history_label, size=(6, 3))
# plt.savefig("acc123.png", dpi=200)
plt.show()

pyabc.visualization.plot_epsilons(history_list, history_label, size=(6, 3))
# plt.savefig("eps123.png", dpi=200)
plt.show()

pyabc.visualization.plot_effective_sample_sizes(history_list, labels=history_label)
plt.show()

w = history_2.get_weighted_distances(t=history_2.max_t)['w']
ess = effective_sample_size(w)

pyabc.visualization.plot_credible_intervals(history_3, size=(8, 24))
plt.show()

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

model_probabilities = history.get_model_probabilities()

max_t = 25
t_id = np.array([i for i in range(max_t)]) + 1

plt.figure(figsize=(15, 5))
plt.bar(x=t_id - 0.2, height=model_probabilities[0][0:max_t], width=0.2, color='tomato')
plt.bar(x=t_id + 0.2, height=model_probabilities[2][0:max_t], width=0.2)
plt.bar(x=t_id, height=model_probabilities[1][0:max_t], width=0.2, color='turquoise')
plt.xlim(0.2, max_t - 0.2, 1)
locs, labels = plt.xticks()
plt.xlabel("Population index")
plt.ylabel("Model probability")
plt.xticks(np.arange(1, max_t, step=1))
plt.legend(['model 3', 'model 4', 'model 5'], bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=False)
plt.savefig("model345cmp.png", dpi=200)
plt.show()

# %% Performance view

db_path = "sqlite:///db/model5_24_more.db"
history = pyabc.History(db_path, _id=1)

history_2 = pyabc.History(db_path, _id=2)

history_1 = pyabc.History(db_path, _id=1)

history_list = [history_1, history_2]
history_label = ['Run A', 'Run B']

print("ID: %d, generations: %d" % (history.id, history.max_t))

all_pop = history.get_all_populations()

duration = np.array(
    [all_pop['population_end_time'][i] - all_pop['population_end_time'][i - 1] for i in np.arange(1, 21, 1)])

for i in range(20):
    duration[i] = duration[i].seconds

pyabc.visualization.plot_sample_numbers(history_list, history_label, title=None, size=(3,4))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
plt.savefig("local_modes_1.png", dpi=200)
plt.show()

plt.bar(range(1, 21), duration)
plt.ylabel('Time (second)')
plt.xticks(np.arange(1, 21, step=1))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
plt.savefig("local_modes_3.png", dpi=200)
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(range(1, 21), all_pop['samples'][1:])
plt.ylabel('Required samples')
plt.xticks(np.arange(1, 21, step=1))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
plt.savefig("local_modes_2.png", dpi=200)
plt.show()

samples = all_pop['samples'][1:]

cl = ['tomato', 'purple'] + ['b']*8 + ['limegreen'] + ['b']*9
plt.figure(figsize=(6, 4))
plt.scatter(x=duration, y=samples, c=cl, alpha=0.7)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
plt.ylabel('Required samples')
plt.xlabel('Execution time')
plt.grid(True)
plt.savefig("local_modes_scatter.png", dpi=200)
# plt.xticks(np.arange(1, 21, step=1))
plt.show()

print(duration)
print(samples / duration)

# %% Posterior

# df, w = history.get_distribution(t=5)
# pyabc.visualization.plot_kde_matrix(df, w)
# plt.show()
#
#
# df, w = history.get_distribution(t=12)
# pyabc.visualization.plot_kde_matrix(df, w)
# plt.show()

result_plot_sp(history, None, para_prior5, 9, 4, savefig=False)

result_plot(history, None, para_prior5, 8, savefig=False)
