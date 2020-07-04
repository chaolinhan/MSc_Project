# Title     : Performance analysis
# Objective : read and analysis performance statistics
# Created by: chaolinhan
# Created on: 2020/7/4

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyabc

# %% Load database

# TODO change database name
db_path = "sqlite:///db/model5_m_log_lp.db"

history = pyabc.History(db_path)

print("ID: %d, generations: %d" % (history.id, history.max_t))

# %% Print statistics

for id in [1, 2, 3]:
    history.id = id
    print(history.total_nr_simulations)
    x = history.get_all_populations()
    print(x["population_end_time"][history.max_t]-x["population_end_time"][0])