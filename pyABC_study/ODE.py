import scipy
import numpy as np


def eqns(var, t0, lambdaN, kNB, muN, vNM, lambdaM, kMB, muM, sBN, iBM, muB, sAM, muA):
    N, M, B, A = var
    dN = lambdaN + kNB * B + muN * N - vNM * N * M
    dM = lambdaM + kMB * B - muM * M
    dB = (sBN * N) / (1 + iBM) - muB * B
    dA = sAM * M - muA * A
    return dN, dM, dB, dA


# Define distance function
# Euclidean distance
def euclidean_distance(dataNormalised, simulation, normalised=False, na_included=True, time_length = 9):
    if dataNormalised.__len__() != simulation.__len__():
        print("Input length Error")
        return

    if not normalised:
        for key in simulation:
            simulation[key] = (simulation[key] - simulation[key].mean()) / simulation[key].std()
    #print(simulation)
    dis = 0.

    for key in dataNormalised:
        for i in range(time_length):
            if not np.isnan(dataNormalised[key][i]):
                dis += np.absolute(pow(dataNormalised[key][i] - simulation[key][i], 2.0))
            # NaN dealing: assume zero discrepancy
            else:
                dis += 0.
        # dis += np.absolute(pow((dataNormalised[key] - simulation[key]), 2.0)).sum()

    return np.sqrt(dis)


class ODESolver:
    # def __init__(self):
        # # Initial states: N, M, B, A
        # self.varInit = v
        # # Time points to solve
        # self.timePoint = t

    varInit = scipy.array([1.8500000, 0.3333333, 1.995670, 0.665976])
    timePoint = scipy.array([0.5, 1, 2, 4, 6, 12, 24, 48, 72])

    def ode_model(self, para):
        sol = scipy.integrate.odeint(
            eqns,
            self.varInit,
            self.timePoint,
            args=(para["lambdaN"], para["kNB"], para["muN"], para["vNM"],
                  para["lambdaM"], para["kMB"], para["muM"],
                  para["sBN"], para["iBM"], para["muB"],
                  para["sAM"], para["muA"]),
        )
        return {"N": sol[:, 0],
                "M": sol[:, 1],
                "B": sol[:, 2],
                "A": sol[:, 3]}
