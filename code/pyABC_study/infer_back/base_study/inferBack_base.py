import os

import pyabc

from pyABC_study.ODE import ODESolver, PriorLimits, para_true1, para_prior

print("\n\n\n Base\n Median eps, 2000 particles, 20 generations\n\n\n")

# %% Get path

ROOT_DIR = os.path.abspath(os.curdir)
db_path = "sqlite:///SMC_base.db"

# %% Generate synthetic data

# True parameters

# para_true = {'iBM': 8.475862809697531,
#              'kMB': 3.7662920313110075,
#              'kNB': 2.2961320437266535,
#              'lambdaM': 8.509867878209329,
#              'lambdaN': 1.5114114729225983,
#              'muA': 5.903807936902964,
#              'muB': 0.38726153092588084,
#              'muM': 3.697974670181216,
#              'muN': 2.6821274451686814,
#              'sAM': 3.62381585701928,
#              'sBN': 3.7176297747866545,
#              'vNM': 0.4248874922862373}

# para_true = {'iBM': 1.0267462374320455,
#              'kMB': 0.07345932286118964,
#              'kNB': 2.359199465995228,
#              'lambdaM': 2.213837884117815,
#              'lambdaN': 7.260925726829641,
#              'muA': 18.94626522780349,
#              'muB': 2.092860392215201,
#              'muM': 0.17722330053184654,
#              'muN': 0.0023917569160019844,
#              'sAM': 10.228522400429998,
#              'sBN': 4.034313992927392,
#              'vNM': 0.3091883041193706}

# Using default time points

solver = ODESolver()
solver.time_point = solver.time_point_default
# obs_data_noisy = solver.ode_model(para_true, flatten=True, add_noise=True)
obs_data_raw = solver.ode_model(para_true1, flatten=True, add_noise=False)

# obs_data_noisy_s = solver.ode_model(para_true1, flatten=False, add_noise=True)
# obs_data_raw_s = solver.ode_model(para_true1, flatten=False, add_noise=False)

print("Target data")
print(obs_data_raw)

# %% Calculate data range as factors:

# range_N = obs_data_raw_s['N'].max() - obs_data_raw_s['N'].min()
# range_M = obs_data_raw_s['M'].max() - obs_data_raw_s['M'].min()
# range_B = obs_data_raw_s['B'].max() - obs_data_raw_s['B'].min()
# range_A = obs_data_raw_s['A'].max() - obs_data_raw_s['A'].min()

# factors = {}

# for i in range(30):
#     factors[i] = 1 / range_N

# for i in range(30, 60):
#     factors[i] = 1 / range_M

# for i in range(60, 90):
#     factors[i] = 1 / range_B

# for i in range(90, 120):
#     factors[i] = 1 / range_A

# scl = 120./sum(factors.values())

# for i in range(120):
#     factors[i] = factors[i] * scl


# %% Plot

# obs_data_plot(solver.timePoint, obs_data_noisy_s, obs_data_raw_s)

# %% Define prior distribution of parameters
# Be careful that RV("uniform", -10, 15) means uniform distribution in [-10, 5], '15' here is the interval length

lim = PriorLimits(1e-6, 10)
lim2 = PriorLimits(0, 1)
lim3 = PriorLimits(0, 10)
# lim2 = PriorLimits(0, 20)
# lim3 = PriorLimits(0, 20)

prior_distribution = "uniform"

para_prior1 = pyabc.Distribution(
    lambda_n=pyabc.RV(prior_distribution, lim3.lb, lim3.interval_length),
    k_n_beta=pyabc.RV(prior_distribution, lim3.lb, lim3.interval_length),
    mu_n=pyabc.RV(prior_distribution, lim2.lb, lim2.interval_length),
    v_n_phi=pyabc.RV(prior_distribution, lim2.lb, lim2.interval_length),

    lambda_phi=pyabc.RV(prior_distribution, lim3.lb, lim3.interval_length),
    k_phi_beta=pyabc.RV(prior_distribution, lim2.lb, lim2.interval_length),
    mu_phi=pyabc.RV(prior_distribution, lim2.lb, lim2.interval_length),

    s_beta_n=pyabc.RV(prior_distribution, lim3.lb, lim3.interval_length),
    i_beta_phi=pyabc.RV(prior_distribution, lim3.lb, lim3.interval_length),
    mu_beta=pyabc.RV(prior_distribution, lim3.lb, lim3.interval_length),

    s_alpha_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_alpha=pyabc.RV(prior_distribution, lim.lb, lim.interval_length)
)

print(prior_distribution)

# para_prior1 = para_prior(lim, prior_distribution, 1)

# %% Define ABC-SMC model

# distanceP2_adpt = pyabc.AdaptivePNormDistance(p=2,
#                                               scale_function=pyabc.distance.root_mean_square_deviation,
#                                               factors=factors
#                                               )
distanceP2 = pyabc.PNormDistance(p=2) #factors=factors)
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

# sampler0 = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=48)

abc = pyabc.ABCSMC(models=solver.ode_model1,
                   parameter_priors=para_prior1,
                   # acceptor=acceptor_adpt,
                   population_size=20,
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
