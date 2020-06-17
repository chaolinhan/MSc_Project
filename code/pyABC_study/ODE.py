import numpy as np
import scipy
from scipy import integrate


def eqns1(var, t0, lambda_n, k_n_phi, mu_n, v_n_phi, lambda_phi, k_phi_beta, mu_phi, s_beta_n, i_beta_phi, mu_beta,
          s_alpha_phi, mu_alpha):
    """
    Model 1 ODEs, 12 parameters
    :param var:
    :param t0:
    :param lambda_n:
    :param k_n_phi:
    :param mu_n:
    :param v_n_phi:
    :param lambda_phi:
    :param k_phi_beta:
    :param mu_phi:
    :param s_beta_n:
    :param i_beta_phi:
    :param mu_beta:
    :param s_alpha_phi:
    :param mu_alpha:
    :return: tuple of the four differentials
    """
    n, phi, beta, alpha = var
    d_n = lambda_n + k_n_phi * beta + mu_n * n - v_n_phi * n * phi
    d_phi = lambda_phi + k_phi_beta * beta - mu_phi * phi
    d_beta = (s_beta_n * n) / (1 + i_beta_phi) - mu_beta * beta
    d_alpha = s_alpha_phi * phi - mu_alpha * alpha
    return d_n, d_phi, d_beta, d_alpha


def arr2d_to_dict(arr: np.ndarray):
    """
    Flatten 2-d array into dict
    :param arr: a 2-d array
    :return: flat dict
    """
    return {i: arr.flatten()[i] for i in range(arr.flatten().__len__())}


def normalise_data(data):
    """
    Normalise the data dictionary
    :param data: numpy array to be normalised
    :return:
    """
    for key in data:
        data[key] = (data[key] - np.nanmean(data[key])) / np.nanstd(data[key])


# def euclidean_distance(dataNormalised, simulation, normalise=False):
#     """
#     Calculate the Euclidean distance of two data
#     Note that un-normalised data must be put as simulation
#     :param dataNormalised: normalised data to be compared with
#     :param simulation: simulation data generated from ODE solver
#     :param normalise: BOOL, indicate to normalise simulation data or not
#     :return: the Euclidean distance
#     """
#     if dataNormalised.__len__() != simulation.__len__():
#         print("Input length Error")
#         return
#
#     if normalise:
#         normalise_data(simulation)
#
#     dis = 0.
#
#     for key in dataNormalised:
#         tmp = (dataNormalised[key] - simulation[key]) ** 2
#         dis += np.nansum(tmp)
#     # for key in dataNormalised:
#     #     for i in range(time_length):
#     #         if not np.isnan(dataNormalised[key][i]):
#     #             dis += pow(dataNormalised[key][i] - simulation[key][i], 2.0)
#     #         # NaN dealing: assume zero discrepancy
#     #         else:
#     #             dis += 0.
#     #     # dis += np.absolute(pow((dataNormalised[key] - simulation[key]), 2.0)).sum()
#
#     return np.sqrt(dis)


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


class ODESolver:
    """
    Solving the ODE system using given `eqns()`
    """
    timePoint_default = np.concatenate(
        (np.array([0., 0.25, 0.5, 1]), np.arange(2, 24, 2), np.arange(24, 74, 4), np.array([96, 120])),
        axis=0)
    timePoint_exp = np.array([0., 0.25, 0.5, 1, 2, 4, 6, 12, 24, 48, 72, 120])
    varInit = np.array([0, 0, 1, 1])

    timePoint = timePoint_default

    def ode_model(self, para, flatten=True, add_noise=True) -> dict:

        """
        Return a list of the ODE results at timePoints
        :param para: parameter of ODEs
        :param flatten: return a flatten dict or not
        :param add_noise: add Gaussian noise to simulated data
        :return: result data in dict format
        """

        # Gaussian distribution for error terms

        sigma_n = 5.66
        sigma_m = 4.59
        sigma_b = 5.15
        sigma_a = 2.42
        mu = 0.
        a = 0.05

        sol = scipy.integrate.odeint(
            eqns,
            self.varInit,
            self.timePoint,
            args=(para["lambdaN"], para["kNB"], para["muN"], para["vNM"],
                  para["lambdaM"], para["kMB"], para["muM"],
                  para["sBN"], para["iBM"], para["muB"],
                  para["sAM"], para["muA"])
        )

        time_len = len(self.timePoint)
        if add_noise:
            sol[1:, 0] += a * sigma_n * np.random.randn(time_len - 1) + mu
            sol[1:, 1] += a * sigma_m * np.random.randn(time_len - 1) + mu
            sol[1:, 2] += a * sigma_b * np.random.randn(time_len - 1) + mu
            sol[1:, 3] += a * sigma_a * np.random.randn(time_len - 1) + mu

        if flatten:
            return {i: sol.flatten()[i] for i in range(sol.flatten().__len__())}
        else:
            return {"N": sol[:, 0],
                    "M": sol[:, 1],
                    "B": sol[:, 2],
                    "A": sol[:, 3]}

    def non_noisy_model(self, para):
        """
        Return a list of the ODE results at timePoints, with no noise
        :param para: parameter of ODEs
        :return: result data in dict format
        """
        return self.ode_model(para, add_noise=False)


class PriorLimits:
    """
    Limits class for parameters' prior range
    """

    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub
        if ub is not None and lb is not None:
            self.interval_length = ub - lb
        else:
            self.interval_length = np.nan
