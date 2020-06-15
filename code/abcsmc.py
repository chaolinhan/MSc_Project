import os
import numpy as np
import pyabc
import pandas as pd
from pyABC_study.ODE import ODESolver, PriorLimits, arr2d_to_dict

print("\n\n\nABC SMC\nParaneter estimation\n")

# %% Set database path

db_path = "sqlite:///abcsmc.db"

# %% Read  and prepare raw data

raw_data_path = os.path.abspath(os.curdir) + "/data/rawData.csv"
raw_data = pd.read_csv(raw_data_path).astype("float32")

time_points: object = raw_data.iloc[:, 0].to_numpy()
exp_data = raw_data.iloc[:, 1:].to_numpy()
exp_data = arr2d_to_dict(exp_data)

exp_data_s = raw_data.iloc[:, 1:].to_dict(orient='list')
for k in exp_data_s:
    exp_data_s[k] = np.array(exp_data_s[k])

print("Target data")
print(exp_data)

solver = ODESolver()
solver.timePoint = time_points

# %% Calculate data range as factors:

print("No factors applied")

# range_N = obs_data_raw_s['N'].max() - obs_data_raw_s['N'].min()
# range_M = obs_data_raw_s['M'].max() - obs_data_raw_s['M'].min()
# range_B = obs_data_raw_s['B'].max() - obs_data_raw_s['B'].min()
# range_A = obs_data_raw_s['A'].max() - obs_data_raw_s['A'].min()
# 
# factors = {}
# 
# for i in range(30):
#     factors[i] = 1 / range_N
# 
# for i in range(30, 60):
#     factors[i] = 1 / range_M
# 
# for i in range(60, 90):
#     factors[i] = 1 / range_B
# 
# for i in range(90, 120):
#     factors[i] = 1 / range_A
# 
# scl = 120./sum(factors.values())
# 
# for i in range(120):
#     factors[i] = factors[i] * scl


# %% Plot

# obs_data_plot(solver.timePoint, obs_data_noisy_s, obs_data_raw_s)

# %% Define prior distribution of parameters
# Be careful that RV("uniform", -10, 15) means uniform distribution in [-10, 5], '15' here is the interval length

lim = PriorLimits(0, 75)

paraPrior = pyabc.Distribution(
    lambdaN=pyabc.RV("uniform", lim.lb, lim.interval_length),
    kNB=pyabc.RV("uniform", lim.lb, lim.interval_length),
    muN=pyabc.RV("uniform", lim.lb, lim.interval_length),
    vNM=pyabc.RV("uniform", lim.lb, lim.interval_length),
    lambdaM=pyabc.RV("uniform", lim.lb, lim.interval_length),
    kMB=pyabc.RV("uniform", lim.lb, lim.interval_length),
    muM=pyabc.RV("uniform", lim.lb, lim.interval_length),
    sBN=pyabc.RV("uniform", lim.lb, lim.interval_length),
    iBM=pyabc.RV("uniform", lim.lb, lim.interval_length),
    muB=pyabc.RV("uniform", lim.lb, lim.interval_length),
    sAM=pyabc.RV("uniform", lim.lb, lim.interval_length),
    muA=pyabc.RV("uniform", lim.lb, lim.interval_length)
)

# %% Define ABC-SMC model

# distanceP2_adpt = pyabc.AdaptivePNormDistance(p=2,
#                                               scale_function=pyabc.distance.root_mean_square_deviation,
#                                               factors=factors
#                                               )
distanceP2 = pyabc.PNormDistance(p=2)#, factors=factors)
# kernel1 = pyabc.IndependentNormalKernel(var=1.0 ** 2)

# Measure distance and set it as minimum epsilon
# min_eps = distanceP2(obs_data_noisy, obs_data_raw)

# acceptor1 = pyabc.StochasticAcceptor()
# acceptor_adpt = pyabc.UniformAcceptor(use_complete_history=True)

eps0 = pyabc.MedianEpsilon(60, median_multiplier=0.9)
# eps1 = pyabc.Temperature()
# eps_fixed = pyabc.epsilon.ListEpsilon([50, 46, 43, 40, 37, 34, 31, 29, 27, 25,
#                                        23, 21, 19, 17, 15, 14, 13, 12, 11, 10])

# transition0 = pyabc.transition.LocalTransition(k=50, k_fraction=None)
# transition1 = pyabc.transition.GridSearchCV()

sampler0 = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=48)

abc = pyabc.ABCSMC(models=solver.non_noisy_model,
                   parameter_priors=paraPrior,
                   # acceptor=acceptor_adpt,
                   population_size=2000,
                   sampler=sampler0,
                   distance_function=distanceP2,
                   # transitions=transition1,
                   eps=eps0,
                   # acceptor=pyabc.UniformAcceptor(use_complete_history=True)
                   )

# %% Print ABC SMC info

print(abc.acceptor)
print(abc.distance_function, abc.distance_function.p)
print(abc.eps)
print(abc.models)
print(abc.population_size, abc.population_size.nr_particles)
print(abc.sampler, abc.sampler.n_procs)
print(abc.transitions)

# %% Run ABC-SMC

abc.new(db_path, exp_data)
max_population = 30
min_eps = 4

print(db_path)
print("Generations: %d" % max_population)
print("Minimum eps: %.3f" % min_eps)


history = abc.run(minimum_epsilon=min_eps, max_nr_populations=max_population)

# %% Plot results

# result_plot(history, para_true, paraPrior, max_population)
# result_plot(history, para_true, paraPrior, history.max_t)
# result_data(history, obs_data_noisy_s, solver.timePoint, history.max_t)
