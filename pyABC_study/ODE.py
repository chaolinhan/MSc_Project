import scipy
from scipy import integrate
import numpy as np


def eqns(var, t0, lambdaN, kNB, muN, vNM, lambdaM, kMB, muM, sBN, iBM, muB, sAM, muA):
    """
The ODE of dynamical system
    :param var: variables to model with
    :param the rest all are model parameters
    :return: ODE status
    """
    N, M, B, A = var
    dN = lambdaN + kNB * B + muN * N - vNM * N * M
    dM = lambdaM + kMB * B - muM * M
    dB = (sBN * N) / (1 + iBM) - muB * B
    dA = sAM * M - muA * A
    return dN, dM, dB, dA


def normalise_data(data):
    """
    Normalise the data dictionary
    :param data: numpy array to be normalised
    :return:
    """
    for key in data:
        data[key] = (data[key] - np.nanmean(data[key])) / np.nanstd(data[key])


def euclidean_distance(dataNormalised, simulation, normalise=False, time_length=9):
    """
    Calculate the Euclidean distance of two data
    Note that un-normalised data must be put as simulation
    :param dataNormalised: normalised data to be compared with
    :param simulation: simulation data generated from ODE solver
    :param normalise: BOOL, indicate to normalise simulation data or not
    :param time_length: number of time points to model with
    :return: the Euclidean distance
    """
    if dataNormalised.__len__() != simulation.__len__():
        print("Input length Error")
        return

    if normalise:
        normalise_data(simulation)

    dis = 0.

    for key in dataNormalised:
        tmp= (dataNormalised[key]-simulation[key])**2
        dis += np.nansum(tmp)
    # for key in dataNormalised:
    #     for i in range(time_length):
    #         if not np.isnan(dataNormalised[key][i]):
    #             dis += pow(dataNormalised[key][i] - simulation[key][i], 2.0)
    #         # NaN dealing: assume zero discrepancy
    #         else:
    #             dis += 0.
    #     # dis += np.absolute(pow((dataNormalised[key] - simulation[key]), 2.0)).sum()

    return np.sqrt(dis)


class ODESolver:

    timepoint_default = np.concatenate((np.array([0., 0.25, 0.5, 1]), np.arange(2, 24, 2), np.arange(24, 74, 4)), axis=0)
    varInit = np.array([0, 0, 0, 0])

    timePoint = timepoint_default

    def ode_model(self, para):
        sol = scipy.integrate.odeint(
            eqns,
            self.varInit,
            self.timePoint,
            args=(para["lambdaN"], para["kNB"], para["muN"], para["vNM"],
                  para["lambdaM"], para["kMB"], para["muM"],
                  para["sBN"], para["iBM"], para["muB"],
                  para["sAM"], para["muA"])
        )
        return {"N": sol[:, 0],
                "M": sol[:, 1],
                "B": sol[:, 2],
                "A": sol[:, 3]}
