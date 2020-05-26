import matplotlib.pyplot as plt
import pandas as pd
import pyabc
import os

from pyABC_study.ODE import ODESolver

rawData_path = os.path.abspath(os.curdir) + "/data/rawData.csv"
rawData = pd.read_csv(rawData_path).astype("float32")

def quantile_calculate(all_data, len, q = 0.5):
    df_quantile = pd.DataFrame(columns=['N', 'M', 'B', 'A'])
    for jj in range(len):
        # print(df_all_sim_data.loc[jj].mean())
        df_quantile = df_quantile.append(all_data.loc[jj].quantile(q), ignore_index=True)
    # print(df_quantile)
    return df_quantile


def sim_data_plot(timePoints, sim_data):
    """
    Plot the data of dict type
    :param timePoints: time points to plot
    :param sim_data: data in duct type
    :return:
    """
    plt.plot(timePoints, sim_data['N'], timePoints, sim_data['M'])
    # plt.scatter(rawData['time'], rawData['N'])
    # plt.scatter(rawData['time'], rawData['M'])
    plt.show()

    plt.plot(timePoints, sim_data['B'], timePoints, sim_data['A'])
    # plt.scatter(rawData['time'], rawData['B'])
    # plt.scatter(rawData['time'], rawData['A'])
    plt.show()


def result_plot(history, nr_population=1):
    """
    Plot the population distribution, eps values and acceptance rate
    :param history: pyABC history object
    :param nr_population: the population to be plotted
    :return:
    """
    pyabc.visualization.plot_acceptance_rates_trajectory(history)
    plt.show()

    pyabc.visualization.plot_epsilons(history)
    plt.show()

    df, w = history.get_distribution(t=nr_population - 1)
    pyabc.visualization.plot_kde_matrix(df, w)

    plt.show()


def result_data(history, compare_data, nr_population=1, sample_size=50):
    df, w = history.get_distribution(t=nr_population - 1)
    df_sample = df.sample(sample_size, replace=False)
    solver = ODESolver()
    df_all_sim_data = pd.DataFrame(columns=['N', 'M', 'B', 'A'])

    for ii in range(sample_size):
        temp_dict = df_sample.iloc[ii].to_dict()
        sim_data = solver.ode_model(temp_dict)
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
    df_25 =  quantile_calculate(df_all_sim_data, solver.timePoint.__len__(), 0.25)
    for kk in range(4):
        fig, ax = plt.subplots()
        ax.plot(solver.timePoint, df_mean.iloc[:,kk], 'r', label="Mean")
        ax.plot(solver.timePoint, df_25.iloc[:,kk], 'b--')
        ax.plot(solver.timePoint, df_75.iloc[:,kk], 'b--')
        index_cov = ['N', 'M', 'B', 'A']
        ax.scatter(solver.timePoint, compare_data[index_cov[kk]])
        ax.legend(['Mean', '25% quantile', '75 quantile', 'Raw'])
        plt.show()