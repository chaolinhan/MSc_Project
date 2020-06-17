import os
import numpy as np
import pyabc
import pandas as pd
from pyABC_study.ODE import ODESolver, PriorLimits, arr2d_to_dict, exp_data, exp_data_s, para_true1
from pyABC_study.dataPlot import result_plot, result_data

print("\n\n\nABC SMC\nParameter estimation\n")

# %% Set database path

# TODO Change database name every run
db_path = "sqlite:///abcsmc.db"

# %% Read  and prepare raw data

print("Target data")
print(exp_data)

solver = ODESolver()
solver.timePoint = solver.timePoint_exp

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

# TODO Set prior

lim = PriorLimits(1e-5, 75)

prior_distribution = "uniform"

print(prior_distribution)

para_prior1 = pyabc.Distribution(
    lambdaN=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    kNB=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    muN=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    vNM=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    lambdaM=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    kMB=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    muM=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    sBN=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    iBM=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    muB=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    sAM=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    muA=pyabc.RV(prior_distribution, lim.lb, lim.interval_length)
)

para_prior2 = pyabc.Distribution(
    lambdaN=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    a=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    kNB=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    muN=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    vNM=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    lambdaM=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    kMB=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    muM=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    sBN=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    iBM=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    muB=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    sAM=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    muA=pyabc.RV(prior_distribution, lim.lb, lim.interval_length)
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

eps0 = pyabc.MedianEpsilon(60)
# eps1 = pyabc.Temperature()
# eps_fixed = pyabc.epsilon.ListEpsilon([50, 46, 43, 40, 37, 34, 31, 29, 27, 25,
#                                        23, 21, 19, 17, 15, 14, 13, 12, 11, 10])

# transition0 = pyabc.transition.LocalTransition(k=50, k_fraction=None)
# transition1 = pyabc.transition.GridSearchCV()

sampler0 = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=48)


abc = pyabc.ABCSMC(models=solver.non_noisy_model1,
                   parameter_priors=para_prior1,
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
print(abc.distance_function.p, abc.distance_function)
print(abc.eps)
print(abc.models)
print(abc.population_size.nr_particles, abc.population_size)
print(abc.sampler.n_procs, abc.sampler)
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

# result_plot(history, None, para_prior1, history.max_t)
#
# solver.ode_model = solver.ode_model1
# result_data(history, exp_data_s, solver, history.max_t)
