# Title     : Performance analysis
# Objective : read and analysis performance statistics
# Created by: chaolinhan
# Created on: 2020/7/4

import os
import numpy as np
import matplotlib.pyplot as plt
import pyabc

# %% Load database

# TODO change database name
db_path = "sqlite:///../dbfiles/model5_36_more.db"

history = pyabc.History(db_path)

# print("ID: %d, generations: %d" % (history.id, history.max_t))

# %% Print statistics

print(db_path.split("/")[-1])

for id in range(1, 11):
    history.id = id
    print("ID: {}".format(history.id))
    print("Sampling size: {}".format(history.total_nr_simulations))
    x = history.get_all_populations()
    print("Execution time: {}".format(x["population_end_time"][history.max_t]-x["population_end_time"][0]))