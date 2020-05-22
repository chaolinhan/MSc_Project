import numpy as np
import os
import pandas as pd
from scipy.optimize import curve_fit
from pyABC_study.ODE import ODESolver, euclidean_distance, normalise_data

ROOT_DIR = os.path.abspath(os.curdir)

rawData_path = ROOT_DIR + "/data/rawData.csv"
rawData = pd.read_csv(rawData_path, header=None)
timePoints = rawData.iloc[:,0].to_numpy()
rawDataArray = rawData.iloc[:,1:].transpose().to_numpy()
rawData = rawData.rename(columns={0:"time", 1:"N", 2:"M", 3:"B", 4:"A"})
rawDataDict = rawData.iloc[:,1:].to_dict(orient = 'series')

paraInit = {"lambdaN": 13.753031, "kNB": 1.581684, "muN": -0.155420, "vNM": 0.262360,
            "lambdaM": 2.993589, "kMB": 0.041040, "muM": 0.201963,
            "sBN": 1.553020, "iBM": -0.046259, "muB": 1.905163,
            "sAM": 11.001731, "muA": 23.022678}

solver = ODESolver()
solver.timePoint = timePoints
expData = solver.ode_model(paraInit)
expDataArray = np.array([expData["N"], expData["M"], expData["B"], expData["A"]])

