# Title     : Sensitivity of resultant models
# Objective : PCA and parameter sensitivity
# Created by: chaolinhan
# Created on: 2020/7/18

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabc
from sklearn.decomposition import PCA

from pyABC_study.ODE import PriorLimits, para_prior

# from pyABC_study.dataPlot import result_data, result_plot

# %% Settings

# change prior
lim = PriorLimits(1e-6, 50)
prior_distribution = "loguniform"

print(prior_distribution)

para_prior1 = para_prior(lim, prior_distribution, 1)
para_prior2 = para_prior(lim, prior_distribution, 2)
para_prior3 = para_prior(lim, prior_distribution, 3)
para_prior4 = para_prior(lim, prior_distribution, 4)
para_prior5 = para_prior(lim, prior_distribution, 5)

# %% Load database

# change database name
db_path = "sqlite:///db/model5_super.db"

history = pyabc.History(db_path)

print("ID: %d, generations: %d" % (history.id, history.max_t))

# %% Load last population

df, w = history.get_distribution(t=history.max_t - 5)

pca = PCA(n_components=12, svd_solver='full', iterated_power='full')
pca.fit(df)

# plt.plot(pca.explained_variance_ratio_)
# plt.show()

plt.figure(figsize=(11, 5))

t_id = np.array([i for i in range(12)]) + 1
#
# plt.bar(x=t_id, height=pca.explained_variance_ratio_, width=0.2)
# plt.show()

plt.bar(t_id, pca.explained_variance_ratio_.cumsum())
plt.ylim(0.5, 1.0, 0.2)
plt.xticks(np.arange(1, 13, step=1))
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(t_id, pca.explained_variance_ratio_)
plt.ylim(0., 0.92, 0.2)
plt.xticks(np.arange(1, 13, step=1))
plt.savefig("all_pc.png", dpi=200)
plt.show()

sum = 0
for j in range(12):
    for i in range(12):
        sum += (pca.components_[i][j]) ** 2
    print(sum)

pca_comp_sq = []

for i in range(4, 12):
    print(pca.components_[i] ** 2)

np.savetxt("pca.csv", np.asarray(pca.components_ ** 2), delimiter=",")