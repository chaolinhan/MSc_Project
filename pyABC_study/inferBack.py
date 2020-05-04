import os
import tempfile

import numpy
import pyabc
import scipy

db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))


# Define ODES

def eqns(var, t0, lambdaN, kNB, muN, vNM, lambdaM, kMB, muM, sBN, iBM, muB, sAM, muA):
    N, M, B, A = var
    dN = lambdaN + kNB * B + muN * N - vNM * N * M
    dM = lambdaM + kMB * B - muM * M
    dB = (sBN * N) / (1 + iBM) - muB * B
    dA = sAM * M - muA * A
    return dN, dM, dB, dA


# Define ODE solver

varInit = scipy.array([1.8500000, 0.3333333, 1.995670, 0.665976])

timePoint = scipy.array([0.5, 1, 2, 4, 6, 12, 24, 48, 72])


def ODEmodel(para):
    sol = scipy.integrate.odeint(
        eqns,
        varInit,
        timePoint,
        args=(para["lambdaN"], para["kNB"], para["muN"], para["vNM"],
              para["lambdaM"], para["kMB"], para["muM"],
              para["sBN"], para["iBM"], para["muB"],
              para["sAM"], para["muA"]),
    )
    return {"N": sol[:, 0],
            "M": sol[:, 1],
            "B": sol[:, 2],
            "A": sol[:, 3]}


# Test ODE solver

paraInit = {"lambdaN": 13.753031, "kNB": 1.581684, "muN": -0.155420, "vNM": 0.262360,
            "lambdaM": 2.993589, "kMB": 0.041040, "muM": 0.201963,
            "sBN": 1.553020, "iBM": -0.046259, "muB": 1.905163,
            "sAM": 11.001731, "muA": 23.022678}

print(ODEmodel(paraInit))

# Normalise the N dimensional data for INPUT

expData = ODEmodel(paraInit)
for key in expData:
    expData[key] = (expData[key] - expData[key].mean()) / expData[key].std()


# Define distance function
# Euclidean distance
def distance(dataNormalised, simulation):

    #
    for key in simulation:
        simulation[key] = (simulation[key] - simulation[key].mean()) / simulation[key].std()
    dis = 0.
    for key in dataNormalised:
        dis += numpy.absolute(pow((dataNormalised[key] - simulation[key]), 2.0)).sum()

    return numpy.sqrt(dis)


# Test distance function

# newData = copy.deepcopy(expData)
# newData["N"][2] = 100
# print(distance(newData, expData))


# Define prior distribution od parameters

paraPrior = pyabc.Distribution(
    lambdaN=pyabc.RV("uniform", 10, 15),
    kNB=pyabc.RV("uniform", 0, 3),
    muN=pyabc.RV("uniform", -1, 1),
    vNM=pyabc.RV("uniform", -1, 1),
    lambdaM=pyabc.RV("uniform", 0, 5),
    kMB=pyabc.RV("uniform", -1, 1),
    muM=pyabc.RV("uniform", -1, 1),
    sBN=pyabc.RV("uniform", 0, 3),
    iBM=pyabc.RV("uniform", -1, 1),
    muB=pyabc.RV("uniform", -1, 5),
    sAM=pyabc.RV("uniform", 8, 15),
    muA=pyabc.RV("uniform", 20, 25)
)

# Define ABC-SMC model

abc = pyabc.ABCSMC(models=ODEmodel,
                   parameter_priors=paraPrior,
                   distance_function=distance,
                   population_size=50,
                   # transitions=pyabc.LocalTransition(k_fraction=.3)
                   eps=pyabc.MedianEpsilon(500, median_multiplier=0.7)
                   )

abc.new(db_path, expData)

history = abc.run(minimum_epsilon=1, max_nr_populations=4)

# print(history.get_distribution(t=2))
history.get_distribution(t=3)[0].to_csv(r"/Users/chaolinhan/OneDrive/PostgraduateProject/pyABC_study/pop3outRaw.csv")