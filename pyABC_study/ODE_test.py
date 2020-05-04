import unittest
from numpy import nan as NaN
import scipy
import numpy as np
from copy import deepcopy
from pyABC_study.ODE import ODESolver, euclidean_distance, normalise_data


paraInit = {"lambdaN": 13.753031, "kNB": 1.581684, "muN": -0.155420, "vNM": 0.262360,
            "lambdaM": 2.993589, "kMB": 0.041040, "muM": 0.201963,
            "sBN": 1.553020, "iBM": -0.046259, "muB": 1.905163,
            "sAM": 11.001731, "muA": 23.022678}

solver = ODESolver()
expData = solver.ode_model(paraInit)
tmp = deepcopy(expData)
normalise_data(tmp)

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)


    def test_solver(self):
        self.assertAlmostEqual(expData["N"].sum(), 79.77811684729392)
        self.assertAlmostEqual(expData["M"].sum(), 89.24608344560379)
        self.assertAlmostEqual(expData["B"].sum(), 63.48068864888955)
        self.assertAlmostEqual(expData["A"].sum(), 42.97632361303949)

    def test_normalise(self):
        for key in tmp:
            self.assertAlmostEqual(tmp[key].mean(),0.)
            self.assertAlmostEqual(tmp[key].std(),1.)

    def test_distance(self):
        tmp_compare = deepcopy(tmp)
        tmp_compare['N'][1] = tmp['N'][1] + 2.
        tmp_compare['M'][2] = tmp['M'][2] + 3.
        tmp_compare['B'][3] = tmp['B'][3] + 1.
        tmp_compare['A'][4] = tmp['A'][4] + 1.
        self.assertAlmostEqual(euclidean_distance(tmp, tmp_compare, normalise=False), np.sqrt(15.))

    def test_distance_NaN(self):
        tmp_na = deepcopy(tmp)
        tmp_na['N'][1] = tmp['N'][1] + 2.
        tmp_na['M'][2] = NaN
        tmp_na['B'][3] = NaN
        tmp_na['A'][4] = tmp['A'][4] + 1.
        self.assertAlmostEqual(euclidean_distance(tmp_na, tmp, normalise=False), np.sqrt(5.))


if __name__ == '__main__':
    unittest.main(verbosity=2)
