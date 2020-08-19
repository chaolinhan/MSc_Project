import os

import pyabc

from pyABC_study.ODE import ODESolver, PriorLimits, para_true1, para_prior

print("\n\n\n Base\n Median eps, 2000 particles, 20 generations\n\n\n")

# %% TODO: Set path

db_path = "sqlite:///dbfiles/ib_wide.db"

# %% Generate synthetic data


# Using default time points

solver = ODESolver()
solver.time_point = solver.time_point_default

obs_data_raw = solver.ode_model1(para_true1)

print("Target data")
print(obs_data_raw)

# TODO: Set factors
# print("Factors applied: 75/25 factor")

# factors = {}

# time_length: int = len(solver.time_point) * 4

# # for i in range(0, time_length, 4):
# #     factors[i] = 1/26.520

# # for i in range(1, time_length, 4):
# #     factors[i] = 1/19.565

# # for i in range(2, time_length, 4):
# #     factors[i] = 1/28.406

# # for i in range(3, time_length, 4):
# #     factors[i] = 1/12.918

# for i in range(int(0.5 * time_length)):
#     factors[i] = 3

# for i in range(int(0.5 * time_length), time_length):
#     factors[i] = 2

# scl = time_length / sum(factors.values())

# for i in range(time_length):
#     factors[i] = factors[i] * scl

# print(factors)

# %% Plot


# %% Define prior distribution of parameters
# Be careful that RV("uniform", -10, 15) means uniform distribution in [-10, 5], '15' here is the interval length

lim = PriorLimits(1e-6, 25)
# lim2 = PriorLimits(1e-6, 1)
# lim3 = PriorLimits(1e-6, 10)
lim2 = PriorLimits(0, 5)
lim3 = PriorLimits(0, 15)

prior_distribution = "uniform"

para_prior1 = pyabc.Distribution(
    lambda_n=pyabc.RV(prior_distribution, lim3.lb, lim3.interval_length),
    k_n_beta=pyabc.RV(prior_distribution, lim3.lb, lim3.interval_length),
    mu_n=pyabc.RV(prior_distribution, lim3.lb, lim3.interval_length),
    v_n_phi=pyabc.RV(prior_distribution, lim2.lb, lim2.interval_length),

    lambda_phi=pyabc.RV(prior_distribution, lim3.lb, lim3.interval_length),
    k_phi_beta=pyabc.RV(prior_distribution, lim2.lb, lim2.interval_length),
    mu_phi=pyabc.RV(prior_distribution, lim2.lb, lim2.interval_length),

    s_beta_n=pyabc.RV(prior_distribution, lim3.lb, lim3.interval_length),
    i_beta_phi=pyabc.RV(prior_distribution, lim3.lb, lim3.interval_length),
    mu_beta=pyabc.RV(prior_distribution, lim2.lb, lim2.interval_length),

    s_alpha_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_alpha=pyabc.RV(prior_distribution, lim.lb, lim.interval_length)
)

print(prior_distribution)

# para_prior1 = para_prior(lim, prior_distribution, 1)

# %% Define ABC-SMC model

# distanceP2_adpt = pyabc.AdaptivePNormDistance(p=2,
#                                               scale_function=pyabc.distance.root_mean_square_deviation
#                                             #   factors=factors
#                                               )
distanceP2 = pyabc.PNormDistance(p=2)#, factors=factors)
# kernel1 = pyabc.IndependentNormalKernel(var=1.0 ** 2)

# Measure distance and set it as minimum epsilon
# min_eps = distanceP2(obs_data_noisy, obs_data_raw)

# acceptor1 = pyabc.StochasticAcceptor()
# acceptor_adpt = pyabc.UniformAcceptor(use_complete_history=True)

eps0 = pyabc.MedianEpsilon(50)
# eps1 = pyabc.Temperature()
# eps_fixed = pyabc.epsilon.ListEpsilon([50, 46, 43, 40, 37, 34, 31, 29, 27, 25,
#                                        23, 21, 19, 17, 15, 14, 13, 12, 11, 10])

# transition0 = pyabc.transition.LocalTransition(k=50, k_fraction=None)
# transition1 = pyabc.transition.GridSearchCV()

# sampler0 = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=1)

abc = pyabc.ABCSMC(models=solver.ode_model1,
                   parameter_priors=para_prior1,
                   # acceptor=acceptor_adpt,
                   population_size=2000,
                   # sampler=sampler0,
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

abc.new(db_path, obs_data_raw)
max_population = 20
min_eps = 4

print(db_path)
print("Generations: %d" % max_population)
print("Minimum eps: %.3f" % min_eps)

history = abc.run(minimum_epsilon=min_eps, max_nr_populations=max_population)

# %% Plot results

# result_plot(history, para_true, paraPrior, max_population)
# result_plot(history, para_true, paraPrior, history.max_t)
# result_data(history, obs_data_noisy_s, solver.timePoint, history.max_t)
