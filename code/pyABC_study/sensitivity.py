# Title     : Sensitivity of resultant models
# Objective : PCA and parameter sensitivity
# Created by: chaolinhan
# Created on: 2020/7/18

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyabc

from pyABC_study.ODE import ODESolver, PriorLimits, arr2d_to_dict, exp_data, exp_data_s, para_prior
from sklearn.decomposition import PCA
# from pyABC_study.dataPlot import result_data, result_plot

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

# %% Load last population

df, w = history.get_distribution(t=history.max_t)

pca = PCA(n_components=12, svd_solver='full', iterated_power='full')
pca.fit(df)

plt.plot(pca.explained_variance_)
plt.show()

for j in range(12):
    sum = 0
    for i in range(6):
        sum += (pca.components_[i][j])**2
    print(sum)

explode = [0.1]*12

labels = df.columns
plt.pie(pca.components_[5]**2, labels=labels, explode=explode)
plt.tight_layout()
plt.show()

for i in range(7, 12):
    print(pca.components_[i]**2)

pyabc.visualization.plot_credible_intervals(history, size=(8, 24))
plt.show()