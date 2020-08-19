import pyabc

from pyABC_study.ODE import ODESolver, PriorLimits, exp_data

print("\n\n\nABC SMC\nParameter estimation\n")

# %% Set database path and observed data

# Change database name every run
db_path = "sqlite:///model1_log.db"

print("Target data")
print(exp_data)

solver = ODESolver()

# %% Calculate data range as factors:

print("No factors applied")


# %% Plot

# obs_data_plot(solver.timePoint, obs_data_noisy_s, obs_data_raw_s)

# %% Define prior distribution of parameters
# Be careful that RV("uniform", -10, 15) means uniform distribution in [-10, 5], '15' here is the interval length

# Set prior

lim = PriorLimits(1e-6, 75)

prior_distribution = "loguniform"

print(prior_distribution)

para_prior1 = pyabc.Distribution(
    lambda_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    k_n_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    v_n_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    lambda_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    k_phi_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    s_beta_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    i_beta_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    s_alpha_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_alpha=pyabc.RV(prior_distribution, lim.lb, lim.interval_length)
)

para_prior2 = pyabc.Distribution(
    lambda_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    a=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    k_n_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    v_n_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    k_phi_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    s_beta_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    i_beta_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    s_alpha_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_alpha=pyabc.RV(prior_distribution, lim.lb, lim.interval_length)
)

para_prior3 = pyabc.Distribution(
    lambda_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    a=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    k_n_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    v_n_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    k_phi_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    s_beta_n=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_beta=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),

    s_alpha_phi=pyabc.RV(prior_distribution, lim.lb, lim.interval_length),
    mu_alpha=pyabc.RV(prior_distribution, lim.lb, lim.interval_length)
)

# %% Define ABC-SMC model

distanceP2 = pyabc.PNormDistance(p=2)  # , factors=factors)

eps0 = pyabc.MedianEpsilon(60)
# eps_fixed = pyabc.epsilon.ListEpsilon([50, 46, 43, 40, 37, 34, 31, 29, 27, 25,
#                                        23, 21, 19, 17, 15, 14, 13, 12, 11, 10])

# transition0 = pyabc.transition.LocalTransition(k=50, k_fraction=None)

# sampler0 = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=48)

abc = pyabc.ABCSMC(models=solver.ode_model1,
                   parameter_priors=para_prior1,
                   population_size=50,
                   # sampler=sampler0,
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

abc.new(db_path, exp_data)
max_population = 30
min_eps = 4

print(db_path)
print("Generations: %d" % max_population)
print("Minimum eps: %.3f" % min_eps)

history = abc.run(minimum_epsilon=min_eps, max_nr_populations=max_population, min_acceptance_rate=1e-4)

# %% Plot results

# result_plot(history, None, para_prior1, history.max_t)
# result_data(history, exp_data_s, solver, history.max_t)
