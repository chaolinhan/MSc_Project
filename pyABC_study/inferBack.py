import os
import tempfile

import pyabc

from pyABC_study.ODE import ODESolver, PriorLimits
from pyABC_study.dataPlot import sim_data_plot, result_plot, result_data

# %% Get path

ROOT_DIR = os.path.abspath(os.curdir)
db_path = "sqlite:///test.db"

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

para_true = {'iBM': 1.0267462374320455,
             'kMB': 0.07345932286118964,
             'kNB': 2.359199465995228,
             'lambdaM': 2.213837884117815,
             'lambdaN': 7.260925726829641,
             'muA': 18.94626522780349,
             'muB': 2.092860392215201,
             'muM': 0.17722330053184654,
             'muN': 0.0023917569160019844,
             'sAM': 10.228522400429998,
             'sBN': 4.034313992927392,
             'vNM': 0.3091883041193706}


# Using default time points
solver = ODESolver()
expData = solver.ode_model(para_true)
expData_no_flatten = solver.ode_model(para_true, flatten=False)
print("Target data")
print(expData)

# %% Plot

sim_data_plot(solver.timePoint, expData_no_flatten)

# %% Define prior distribution of parameters
# Be careful that RV("uniform", -10, 15) means uniform distribution in [-10, 5], '15' here is the interval length

lim = PriorLimits(0, 20)
lim2 = PriorLimits(0, 1)
lim3 = PriorLimits(0, 10)

paraPrior = pyabc.Distribution(
    lambdaN=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    kNB=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    muN=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
    vNM=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
    lambdaM=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    kMB=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
    muM=pyabc.RV("uniform", lim2.lb, lim2.interval_length),
    sBN=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    iBM=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    muB=pyabc.RV("uniform", lim3.lb, lim3.interval_length),
    sAM=pyabc.RV("uniform", lim.lb, lim.interval_length),
    muA=pyabc.RV("uniform", lim.lb, lim.interval_length)
)

# %% Define ABC-SMC model

distanceP2_adaptive = pyabc.AdaptivePNormDistance(p=2,
                                                  scale_function=pyabc.distance.root_mean_square_deviation
                                                  )
distanceP2 = pyabc.PNormDistance(p=2)
acceptor1 = pyabc.StochasticAcceptor()
eps0 = pyabc.MedianEpsilon(100)
eps1 = pyabc.Temperature()
kernel1 = pyabc.IndependentNormalKernel(var=1.0**2)

abc = pyabc.ABCSMC(models=solver.ode_model,
                   parameter_priors=paraPrior,
                   # acceptor=acceptor1,
                   population_size=100,
                   distance_function=distanceP2,
                   eps=eps0,
                   # acceptor=pyabc.UniformAcceptor(use_complete_history=True)
                   )


# %% Run ABC-SMC

abc.new(db_path, expData)
max_population = 15
history = abc.run(minimum_epsilon=5, max_nr_populations=max_population)


# %% Plot results

# result_plot(history, para_true, paraPrior, max_population)
result_plot(history, para_true, paraPrior, history.max_t)
result_data(history, expData_no_flatten, solver.timePoint, history.max_t)
