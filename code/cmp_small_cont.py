import pyabc

from pyABC_study.ODE import ODESolver, PriorLimits, exp_data, para_prior

print("\n\n\nABC SMC\nParameter estimation\n")

# %% Set database path and observed data

# TODO: Change database name every run
db_path = "sqlite:///dbfiles/model345_cmp_5000.db"

print("Target data")
print(exp_data)

solver = ODESolver()

# %% Calculate data range as factors:

# TODO: Set factors
# print("Factors applied: first 32 data points are more important")

# factors = {}

# for i in range(32):
#     factors[i] = 0.75

# for i in range(32, 48):
#     factors[i] = 0.25


# scl = 48/sum(factors.values())

# for i in range(48):
#    factors[i] = factors[i] * scl


# %% Define prior distribution of parameters
# Be careful that RV("uniform", -10, 15) means uniform distribution in [-10, 5], '15' here is the interval length

# TODO: Set prior

lim = PriorLimits(1e-6, 50)

prior_distribution = "loguniform"

print(prior_distribution)

para_prior1 = para_prior(lim, prior_distribution, 1)
para_prior2 = para_prior(lim, prior_distribution, 2)
para_prior3 = para_prior(lim, prior_distribution, 3)
para_prior4 = para_prior(lim, prior_distribution, 4)
para_prior5 = para_prior(lim, prior_distribution, 5)


# %% Define ABC-SMC model

distanceP2 = pyabc.PNormDistance(p=2)#, factors=factors)

eps0 = pyabc.MedianEpsilon(21.20380683456598)
# eps_fixed = pyabc.epsilon.ListEpsilon([50, 46, 43, 40, 37, 34, 31, 29, 27, 25,
#                                        23, 21, 19, 17, 15, 14, 13, 12, 11, 10])

# transition0 = pyabc.transition.LocalTransition(k=50, k_fraction=None)

# sampler0 = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=6)

population0 = 5000
# population_adpt = pyabc.populationstrategy.AdaptivePopulationSize(start_nr_particles=population0, mean_cv=0.10,
#                                                                   max_population_size=50)

# TODO: set model and prior
abc = pyabc.ABCSMC(models=[solver.ode_model3, solver.ode_model4, solver.ode_model5],
                   parameter_priors=[para_prior3, para_prior4, para_prior5],
                   population_size=population0,
                #    sampler=sampler0,
                   distance_function=distanceP2,
                   eps=eps0,
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

# abc.new(db_path, exp_data)
abc.load(db_path, 1)
max_population = 10
min_eps = 4

print("\n"+db_path+"\n")
print("Generations: %d" % max_population)
print("Minimum eps: %.3f" % min_eps)

history = abc.run(minimum_epsilon=min_eps, max_nr_populations=max_population)

# %% Plot results

# result_plot(history, None, para_prior1, history.max_t)
# result_data(history, exp_data_s, solver, history.max_t)