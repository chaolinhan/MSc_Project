from pyabc import (ABCSMC,
                   RV, Distribution,
                   MedianEpsilon,
                   LocalTransition)
from pyabc.visualization import plot_kde_2d, plot_data_callback
import matplotlib.pyplot as plt
import os
import tempfile
import numpy as np
import scipy

db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))


def eqns(var, t0, lambdaN, kNB, muN, vNM, lambdaM, kMB, muM, sBN, iBM, muB, sAM, muA):
    N, M, B, A = var
    dN = lambdaN + kNB * B + muN * N - vNM * N * M
    dM = lambdaM + kMB * B - muM * M
    dB = (sBN * N) / (1 + iBM) - muB * B
    dA = sAM * M - muA * A
    return dN, dM, dB, dA


paraInit = scipy.array([1.8500000, 0.3333333, 1.995670, 0.665976])

timePoint = scipy.array([0.5, 1, 2, 4, 6, 12, 24, 48, 72])

def ODEmodel(para):
    sol = scipy.integrate.odeint(
        eqns,
        paraInit,
        timePoint,
        args = (para["lambdaN"], para["kNB"], para["mumN"], para["vNM"],
                para["lambdaM"], para["kMB"], para["muM"],
                para["sBN"],para["iBM"],para["muB"],
                para["sAM"],para["muA"])
    )
    return sol


