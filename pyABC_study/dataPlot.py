import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabc

from ODE import ODESolver

rawData_path = os.path.abspath(os.curdir) + "/data/rawData.csv"
rawData = pd.read_csv(rawData_path).astype("float32")


def quantile_calculate(all_data, length, q=0.5):
    """
Calculate quantiles
    :param all_data: all data
    :param length: time length
    :param q: quantile
    :return: data frame of the quantile
    """
    df_quantile = pd.DataFrame(columns=['N', 'M', 'B', 'A'])
    for jj in range(length):
        # print(df_all_sim_data.loc[jj].mean())
        df_quantile = df_quantile.append(all_data.loc[jj].quantile(q), ignore_index=True)
    # print(df_quantile)
    return df_quantile


def obs_data_plot(time_points: np.array, obs_data_noisy, obs_data_raw):
    """
    Plot the data of dict type
    :param time_points: time points to plot
    :param obs_data_noisy: data in duct type
    :return:
    """
    plt.plot(time_points, obs_data_noisy['N'], linestyle=':', label='Noisy N')
    plt.plot(time_points, obs_data_noisy['M'], linestyle=':', label='Noisy Phi')
    if obs_data_raw is not None:
        plt.plot(time_points, obs_data_raw['N'], alpha=0.5, label='Raw N')
        plt.plot(time_points, obs_data_raw['M'], alpha=0.5, label='Raw Phi')
    plt.legend()
    plt.show()

    plt.plot(time_points, obs_data_noisy['B'], linestyle=':', label='Noisy beta')
    plt.plot(time_points, obs_data_noisy['A'], linestyle=':', label='Noisy alpha')
    if obs_data_raw is not None:
        plt.plot(time_points, obs_data_raw['B'], alpha=0.5, label='Raw beta')
        plt.plot(time_points, obs_data_raw['A'], alpha=0.5, label='Raw alpha')
    plt.legend()
    plt.show()


def result_plot(history, true_parameter: dict, limits: pyabc.Distribution, nr_population=1):
    """
Plot the population distribution, eps values and acceptance rate
    :param limits: Limits of the plot
    :param true_parameter: true parameter
    :param history: pyABC history object
    :param nr_population: the population to be plotted
    :return:
    """
    pyabc.visualization.plot_acceptance_rates_trajectory(history)
    plt.show()

    pyabc.visualization.plot_epsilons(history)
    plt.show()

    df, w = history.get_distribution(t=nr_population - 1)

    # Parameters in the first equation

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    idx = 0
    for keys in ['lambdaN', 'kNB', 'muN', 'vNM']:
        # print(keys+": %.2f" % true_parameter[keys])
        pyabc.visualization.plot_kde_1d(df, w, x=keys, ax=ax[idx], xmin=limits[keys].args[0],
                                        xmax=limits[keys].args[0] + limits[keys].args[1])
        ax[idx].axvline(true_parameter[keys], color='r', linestyle='dashed',
                        label="True value\n%.3f" % true_parameter[keys])
        ax[idx].axvspan(df[keys].quantile(0.25), df[keys].quantile(0.75), color="b", alpha=0.5,
                        label="Inter-quartile\n[%.3f, %.3f]\nMean %.3f" % (
                            df[keys].quantile(0.25), df[keys].quantile(0.75), df[keys].mean()))
        ax[idx].legend()
        idx += 1
    fig.suptitle('ODE 1: dN/dt')
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    idx = 0
    for keys in ['lambdaM', 'kMB', 'muM']:
        pyabc.visualization.plot_kde_1d(df, w, x=keys, ax=ax[idx], xmin=limits[keys].args[0],
                                        xmax=limits[keys].args[0] + limits[keys].args[1])
        ax[idx].axvline(true_parameter[keys], color='r', linestyle='dashed',
                        label="True value\n%.3f" % true_parameter[keys])
        ax[idx].axvspan(df[keys].quantile(0.25), df[keys].quantile(0.75), color="b", alpha=0.5,
                        label="Inter-quartile\n[%.3f, %.3f]\nMean %.3f" % (
                            df[keys].quantile(0.25), df[keys].quantile(0.75), df[keys].mean()))
        ax[idx].legend()
        idx += 1
    fig.suptitle('ODE 2: d(Phi)/dt')
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    idx = 0
    for keys in ['sBN', 'iBM', 'muB']:
        pyabc.visualization.plot_kde_1d(df, w, x=keys, ax=ax[idx], xmin=limits[keys].args[0],
                                        xmax=limits[keys].args[0] + limits[keys].args[1])
        ax[idx].axvline(true_parameter[keys], color='r', linestyle='dashed',
                        label="True value\n%.3f" % true_parameter[keys])
        ax[idx].axvspan(df[keys].quantile(0.25), df[keys].quantile(0.75), color="b", alpha=0.5,
                        label="Inter-quartile\n[%.3f, %.3f]\nMean %.3f" % (
                            df[keys].quantile(0.25), df[keys].quantile(0.75), df[keys].mean()))
        ax[idx].legend()
        idx += 1
    fig.suptitle('ODE 3: d(beta)/dt')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    idx = 0
    for keys in ['sAM', 'muA']:
        pyabc.visualization.plot_kde_1d(df, w, x=keys, ax=ax[idx], xmin=limits[keys].args[0],
                                        xmax=limits[keys].args[0] + limits[keys].args[1])
        ax[idx].axvline(true_parameter[keys], color='r', linestyle='dashed',
                        label="True value\n%.3f" % true_parameter[keys])
        ax[idx].axvspan(df[keys].quantile(0.25), df[keys].quantile(0.75), color="b", alpha=0.5,
                        label="Inter-quartile\n[%.3f, %.3f]\nMean %.3f" % (
                            df[keys].quantile(0.25), df[keys].quantile(0.75), df[keys].mean()))
        ax[idx].legend()
        idx += 1
    fig.suptitle('ODE 4: d(alpha)/dt')
    plt.show()

    # Parameters in the second equation
    #
    # fig2 = plt.figure(figsize=(12, 4))
    # idx = 1
    # for keys in ['lambdaM', 'kMB', 'muM']:
    #     ax = fig2.add_subplot(1, 3, idx)
    #     pyabc.visualization.plot_kde_1d(df, w, x=keys, ax=ax)
    #     ax.axvline(true_parameter[keys], color='k', linestyle='dashed', label="True value")
    #     ax.legend()
    #     idx += 1
    # fig2.suptitle('ODE 2: d𝛷/dt')
    # plt.show()
    #
    pyabc.visualization.plot_kde_matrix(df, w)
    plt.show()


def result_data(history, compare_data, time_points, nr_population=1, sample_size=50):
    """
Visualise SMC population and compare it with target data
    :param history: abc.history object
    :param compare_data: target data
    :param nr_population: the id of pupolation to be visualised
    :param sample_size: sampling size of the selected population
    """
    df, w = history.get_distribution(t=nr_population - 1)
    df_sample = df.sample(sample_size, replace=False)
    solver = ODESolver()
    solver.timePoint = time_points
    df_all_sim_data = pd.DataFrame(columns=['N', 'M', 'B', 'A'])

    for ii in range(sample_size):
        temp_dict = df_sample.iloc[ii].to_dict()
        sim_data = solver.ode_model(temp_dict, flatten=False)
        # print(sim_data)
        sim_data = pd.DataFrame.from_dict(sim_data)
        df_all_sim_data = pd.concat([df_all_sim_data, sim_data])

    # plt.ylim(0, 2) # TODO make lim a function parameter
    # plt.show()

    df_mean = pd.DataFrame(columns=['N', 'M', 'B', 'A'])
    for jj in range(solver.timePoint.__len__()):
        # print(df_all_sim_data.loc[jj].mean())
        df_mean = df_mean.append(df_all_sim_data.loc[jj].mean(), ignore_index=True)

    df_median = quantile_calculate(df_all_sim_data, solver.timePoint.__len__(), 0.5)
    df_75 = quantile_calculate(df_all_sim_data, solver.timePoint.__len__(), 0.75)
    df_25 = quantile_calculate(df_all_sim_data, solver.timePoint.__len__(), 0.25)

    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    for kk in range(4):
        axs[kk].plot(solver.timePoint, df_mean.iloc[:, kk], 'r', label="Mean", alpha=0.6)
        # axs[kk].plot(solver.timePoint, df_25.iloc[:, kk], 'b--')
        # axs[kk].plot(solver.timePoint, df_75.iloc[:, kk], 'b--')
        axs[kk].fill_between(solver.timePoint, df_25.iloc[:, kk], df_75.iloc[:, kk], alpha=0.5)
        index_cov = ['N', 'M', 'B', 'A']
        axs[kk].scatter(solver.timePoint, compare_data[index_cov[kk]], alpha=0.7)
        axs[kk].legend(['Mean', '25% – 75% quantile range', 'Observed'])
        axs[kk].set_title(index_cov[kk])
    plt.show()
