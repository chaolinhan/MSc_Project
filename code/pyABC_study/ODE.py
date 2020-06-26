import numpy as np
import scipy
from scipy import integrate


class ODESolver:
    """
    Solving the ODE system using given `eqns()`
    """

    # def __init__(self, eqns):
    #     self.eqns = eqns

    time_point_default = np.concatenate(
        (np.array([0., 0.25, 0.5, 1]), np.arange(2, 24, 2), np.arange(24, 74, 4), np.array([96, 120])),
        axis=0)
    time_point_exp = np.array([0., 0.25, 0.5, 1, 2, 4, 6, 12, 24, 48, 72, 120])
    var_init = np.array([0, 0, 1, 1])

    time_point = time_point_exp

    def ode_model1(self, para, flatten=True, add_noise=False) -> dict:
        """
        Return a list of the ODE results at timePoints
        :param para: parameter of ODEs
        :param flatten: return a flatten dict or not
        :param add_noise: add Gaussian noise to simulated data
        :return: result data in dict format
        """

        # Gaussian distribution for error terms
        # TODO fix std
        # sigma_n = 5.66
        # sigma_m = 4.59
        # sigma_b = 5.15
        # sigma_a = 2.42
        # mu = 0.
        # a = 0.05

        # def eqns1(var, t0, lambda_n, k_n_beta, mu_n, v_n_phi, lambda_phi, k_phi_beta, mu_phi, s_beta_n, i_beta_phi,
        #           mu_beta,
        #           s_alpha_phi, mu_alpha):

        sol = scipy.integrate.odeint(
            eqns1,
            self.var_init,
            self.time_point,
            args=(para["lambda_n"], para["k_n_beta"], para["mu_n"], para["v_n_phi"],
                  para["lambda_phi"], para["k_phi_beta"], para["mu_phi"],
                  para["s_beta_n"], para["i_beta_phi"], para["mu_beta"],
                  para["s_alpha_phi"], para["mu_alpha"])
        )

        time_len = len(self.time_point)

        # if add_noise:
        #     sol[1:, 0] += a * sigma_n * np.random.randn(time_len - 1) + mu
        #     sol[1:, 1] += a * sigma_m * np.random.randn(time_len - 1) + mu
        #     sol[1:, 2] += a * sigma_b * np.random.randn(time_len - 1) + mu
        #     sol[1:, 3] += a * sigma_a * np.random.randn(time_len - 1) + mu

        if flatten:
            return {i: sol.flatten()[i] for i in range(sol.flatten().__len__())}
        else:
            return {"N": sol[:, 0],
                    "M": sol[:, 1],
                    "B": sol[:, 2],
                    "A": sol[:, 3]}

    def ode_model2(self, para, flatten=True, add_noise=False) -> dict:
        """
        Return a list of the ODE results at timePoints
        :param para: parameter of ODEs
        :param flatten: return a flatten dict or not
        :param add_noise: add Gaussian noise to simulated data
        :return: result data in dict format
        """

        # Gaussian distribution for error terms
        # TODO fix std
        # sigma_n = 5.66
        # sigma_m = 4.59
        # sigma_b = 5.15
        # sigma_a = 2.42
        # mu = 0.
        # a = 0.05

        sol = scipy.integrate.odeint(
            eqns2,
            self.var_init,
            self.time_point,
            args=(para["lambda_n"], para["a"], para["k_n_beta"], para["mu_n"], para["v_n_phi"],
                  para["k_phi_beta"], para["mu_phi"],
                  para["s_beta_n"], para["i_beta_phi"], para["mu_beta"],
                  para["s_alpha_phi"], para["mu_alpha"])
        )

        time_len = len(self.time_point)

        # if add_noise:
        #     sol[1:, 0] += a * sigma_n * np.random.randn(time_len - 1) + mu
        #     sol[1:, 1] += a * sigma_m * np.random.randn(time_len - 1) + mu
        #     sol[1:, 2] += a * sigma_b * np.random.randn(time_len - 1) + mu
        #     sol[1:, 3] += a * sigma_a * np.random.randn(time_len - 1) + mu

        if flatten:
            return {i: sol.flatten()[i] for i in range(sol.flatten().__len__())}
        else:
            return {"N": sol[:, 0],
                    "M": sol[:, 1],
                    "B": sol[:, 2],
                    "A": sol[:, 3]}

    def ode_model3(self, para, flatten=True, add_noise=False) -> dict:
        """
        Return a list of the ODE results at timePoints
        :param para: parameter of ODEs
        :param flatten: return a flatten dict or not
        :param add_noise: add Gaussian noise to simulated data
        :return: result data in dict format
        """

        # Gaussian distribution for error terms
        # TODO fix std
        # sigma_n = 5.66
        # sigma_m = 4.59
        # sigma_b = 5.15
        # sigma_a = 2.42
        # mu = 0.
        # a = 0.05

        # def eqns1(var, t0, lambda_n, k_n_beta, mu_n, v_n_phi, lambda_phi, k_phi_beta, mu_phi, s_beta_n, i_beta_phi,
        #           mu_beta,
        #           s_alpha_phi, mu_alpha):

        sol = scipy.integrate.odeint(
            eqns3,
            self.var_init,
            self.time_point,
            args=(para["lambda_n"], para["a"], para["k_n_beta"], para["mu_n"], para["v_n_phi"],
                  para["k_phi_beta"], para["mu_phi"],
                  para["s_beta_n"], para["mu_beta"],
                  para["s_alpha_phi"], para["mu_alpha"])
        )

        time_len = len(self.time_point)

        # if add_noise:
        #     sol[1:, 0] += a * sigma_n * np.random.randn(time_len - 1) + mu
        #     sol[1:, 1] += a * sigma_m * np.random.randn(time_len - 1) + mu
        #     sol[1:, 2] += a * sigma_b * np.random.randn(time_len - 1) + mu
        #     sol[1:, 3] += a * sigma_a * np.random.randn(time_len - 1) + mu

        if flatten:
            return {i: sol.flatten()[i] for i in range(sol.flatten().__len__())}
        else:
            return {"N": sol[:, 0],
                    "M": sol[:, 1],
                    "B": sol[:, 2],
                    "A": sol[:, 3]}

    ode_model = ode_model1


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


def arr2d_to_dict(arr: np.ndarray):
    """
    Flatten 2-d array into dict
    :param arr: a 2-d array
    :return: flat dict
    """
    return {i: arr.flatten()[i] for i in range(arr.flatten().__len__())}


para_true1 = {'i_beta_phi': 1.0267462374320455,
              'k_phi_beta': 0.07345932286118964,
              'k_n_beta': 2.359199465995228,
              'lambda_phi': 2.213837884117815,
              'lambda_n': 7.260925726829641,
              'mu_alpha': 18.94626522780349,
              'mu_beta': 2.092860392215201,
              'mu_phi': 0.17722330053184654,
              'mu_n': 0.0023917569160019844,
              's_alpha_phi': 10.228522400429998,
              's_beta_n': 4.034313992927392,
              'v_n_phi': 0.3091883041193706}

exp_data = {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: np.nan, 5: np.nan, 6: 1.8725, 7: 0.96225, 8: 1.85, 9: 0.33333334,
            10: np.nan, 11: np.nan, 12: 10.708333, 13: 1.08, 14: 4.01255, 15: 1.0565, 16: 26.52, 17: 3.275862,
            18: np.nan, 19: np.nan, 20: 22.333334, 21: 4.8333335, 22: 29.17, 23: 13.88, 24: 16.6, 25: 12.652174,
            26: 7.365, 27: 8.6005, 28: np.nan, 29: np.nan, 30: 5.985, 31: 7.4845, 32: 6.8076925, 33: 18.766666,
            34: 8.76, 35: 3.060075, 36: 5.851852, 37: 19.565218, 38: 1.076415, 39: 6.56325, 40: 0.5555556,
            41: 14.275862, 42: 0.764, 43: 5.7875, 44: 0.3448276, 45: 8.653846, 46: np.nan, 47: np.nan}

exp_data_s = {'N': np.array([0., np.nan, 1.85000002, 10.70833302, 26.52000046,
                             22.33333397, 16.60000038, np.nan, 6.80769253, 5.85185194,
                             0.55555558, 0.34482759]),
              'M': np.array([0., np.nan, 0.33333334, 1.08000004, 3.27586198,
                             4.83333349, 12.652174, np.nan, 18.76666641, 19.56521797,
                             14.27586174, 8.65384579]),
              'B': np.array([1., 1.87249994, np.nan, 4.01254988, np.nan,
                             29.17000008, 7.36499977, 5.98500013, 8.76000023, 1.07641494,
                             0.764, np.nan]),
              'A': np.array([1., 0.96224999, np.nan, 1.05649996, np.nan,
                             13.88000011, 8.60050011, 7.48449993, 3.06007504, 6.56325006,
                             5.7874999, np.nan])}

exp_data_SEM = {'N': ([0., np.nan, 0.31850803, 0.53662652, 1.61257553,
                       1.11068368, 1.08012342, np.nan, 0.69507992, 0.53296918,
                       0.15408339, 0.11399445]),
                'M': ([0., np.nan, 0.14328782, 0.17243356, 0.56563437,
                       0.71643913, 1.4454627, np.nan, 0.92684525, 1.31700051,
                       0.97996753, 1.17792821]),
                'B': ([0., 0.47389999, np.nan, 1.03059006, np.nan,
                       13.74499989, 2.68099999, 2.17700005, 3.2105, 0.33115,
                       3.62849998, np.nan]),
                'A': ([0., 0.41679999, np.nan, 0.45993999, np.nan,
                       6.09899998, 4.8125, 4.18849993, 1.14256001, 3.19499993,
                       7.12750006, np.nan])}


def eqns1(var, t0, lambda_n, k_n_beta, mu_n, v_n_phi, lambda_phi, k_phi_beta, mu_phi, s_beta_n, i_beta_phi, mu_beta,
          s_alpha_phi, mu_alpha):
    n, phi, beta, alpha = var
    d_n = lambda_n + k_n_beta * beta - mu_n * n - v_n_phi * n * phi
    d_phi = lambda_phi + k_phi_beta * beta - mu_phi * phi
    d_beta = (s_beta_n * n) / (1 + i_beta_phi * phi) - mu_beta * beta
    d_alpha = s_alpha_phi * phi - mu_alpha * alpha
    return d_n, d_phi, d_beta, d_alpha


def eqns2(var, t0, lambda_n, a, k_n_beta, mu_n, v_n_phi, k_phi_beta, mu_phi, s_beta_n, i_beta_phi, mu_beta,
          s_alpha_phi, mu_alpha):
    """
    Exponential decay lambda_n
    """
    n, phi, beta, alpha = var
    d_n = lambda_n * np.exp(-a * t0) + k_n_beta * beta - mu_n * n - v_n_phi * n * phi
    d_phi = k_phi_beta * beta - mu_phi * phi
    d_beta = (s_beta_n * n) / (1 + i_beta_phi * phi) - mu_beta * beta
    d_alpha = s_alpha_phi * phi - mu_alpha * alpha
    return d_n, d_phi, d_beta, d_alpha


def eqns3(var, t0, lambda_n, a, k_n_beta, mu_n, v_n_phi, k_phi_beta, mu_phi, s_beta_n, mu_beta,
          s_alpha_phi, mu_alpha):
    """
    No (i_beta_phi * phi) term
    """
    n, phi, beta, alpha = var
    d_n = lambda_n * np.exp(-a * t0) + k_n_beta * beta - mu_n * n - v_n_phi * n * phi
    d_phi = k_phi_beta * beta - mu_phi * phi
    d_beta = s_beta_n * n - mu_beta * beta
    d_alpha = s_alpha_phi * phi - mu_alpha * alpha
    return d_n, d_phi, d_beta, d_alpha

# def normalise_data(data):
#     """
#     Normalise the data dictionary
#     :param data: numpy array to be normalised
#     :return:
#     """
#     for key in data:
#         data[key] = (data[key] - np.nanmean(data[key])) / np.nanstd(data[key])


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
