import matplotlib.pyplot as plt
import pandas as pd
import pyabc
import os

from pyABC_study.ODE import ODESolver

rawData_path = os.path.abspath(os.curdir) + "/data/rawData.csv"
rawData = pd.read_csv(rawData_path).astype("float32")


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


def result_data(history, nr_population=1, sample_size=20):
    df, w = history.get_distribution(t=nr_population - 1)
    df_sample = df.sample(sample_size, replace=False)
    solver = ODESolver()
    df_all_sim_data = pd.DataFrame(columns=['N', 'M', 'B', 'A'])
    for ii in range(sample_size):
        temp_dict = df_sample.iloc[ii].to_dict()
        sim_data = solver.ode_model(temp_dict)
        print(sim_data)
        # sim_data = pd.DataFrame.from_dict(sim_data)
        # df_all_sim_data = pd.concat([df_all_sim_data, sim_data])
        plt.plot(solver.timePoint, sim_data['N'])
        plt.ylim(0, 20) # TODO make lim a function parameter
        plt.show()
    # df_mean = pd.DataFrame(columns=['N', 'M', 'B', 'A'])
    # for jj in range(solver.timePoint.__len__()):
    #     # print(df_all_sim_data.loc[jj].mean())
    #     df_mean = df_mean.append(df_all_sim_data.loc[jj].mean(), ignore_index=True)
    #     print(df_mean)
    #
    # plt.plot(solver.timePoint, df_mean.iloc[:,0], solver.timePoint, df_mean.iloc[:,1])
    # plt.show()



# Make a gif

# limits = dict(
#     lambdaN=(-10, 15),
#     kNB=(-15, 2),
#     muN=(-15, 2),
#     vNM=(-23, 2),
#     lambdaM=(-3, 7),
#     kMB=(-15, 2),
#     muM=(-3, 4),
#     sBN=(-15, 2),
#     iBM=(-15, 2),
#     muB=(0, 12),
#     sAM=(-10, 30),
#     muA=(-20, 80)
#               )
#
#
# # TODO make file name dynamic
#
# for tt in range(1,max_population-1):
#     filename = ROOT_DIR+"/pyABC_study/plot/p500e50t25/"+str(tt)+".png"
#     df, w = history.get_distribution(t=tt)
#     pyabc.visualization.plot_kde_matrix(df, w, limits=limits)
#     plt.savefig(filename)


# Resume

# abc_continue = pyabc.ABCSMC(models=solver.ode_model,
#                    parameter_priors=paraPrior,
#                    distance_function=euclidean_distance,
#                    #distance_function=distance_adaptive,
#                    population_size=1000,
#                    eps=pyabc.MedianEpsilon(30, median_multiplier=1)
#                    )
#
# abc_continue.load(db_path, 3)
