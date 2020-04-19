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
              para["sAM"], para["muA"])
    )
    return {"N": sol[:,0],
            "M": sol[:,1],
            "A": sol[:,2],
            "B": sol[:,3]}


# Test ODE solver

paraInit = {"lambdaN": 13.753031, "kNB": 1.581684, "muN": -0.155420, "vNM": 0.262360,
            "lambdaM": 2.993589, "kMB": 0.041040, "muM": 0.201963,
            "sBN": 1.553020, "iBM": -0.046259, "muB": 1.905163,
            "sAM": 11.001731, "muA": 23.022678}

print(ODEmodel(paraInit))




