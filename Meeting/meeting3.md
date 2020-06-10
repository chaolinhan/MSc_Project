# Measurement noise

![image-20200609170943091](../../../Library/Application Support/typora-user-images/image-20200609170943091.png)

​																												Figure 1

-   (a): add noise to observed data and model (simulator)
-   (b): add noise to observed data
-   (c): no noise added



Adding noise makes the inference harder. Later implementations we could consider using 

-   A model with noise, or
-   [Stochastic acceptor](https://pyabc.readthedocs.io/en/latest/examples/noise.html) provided by `pyabc`

There are also papers using synthetic data with noise and model without noise to test the performance of ABC SMC

# Kernels - efficiency of algorithm

Goal: find the most efficient kernels under a given schedule.

Give a target threshold eps_t, the efficiency can be measure by

1.  Number of required sample to reach eps_t
2.  Acceptance rates and effective sampling size in each generations

Relayed paper:

>   Filippi, S., Barnes, C. P., Cornebise, J., & Stumpf, M. P. H. (2013). On optimality of kernels for approximate Bayesian computation using sequential Monte Carlo. *Statistical Applications in Genetics and Molecular Biology*, *12*(1), 87–107. https://doi.org/10.1515/sagmb-2012-0069

There is limited options in perturbation provided by `pyabc `. Other packages may provide more kernel options

Kernels provided by `pyabc`:

-   Multivariate normal kernel
    -   One parameter: `scaling` default to 1
    -   Scaling is a factor which additionally multiplies the covariance with
-   Multivariate normal kernel with M nearest neighbours 
    -   For each particle, M-nearest neighbours are selected, the covariance is calculated using these neighbours 

## Experiments

### Fixed threshold schedule

Under a fixed threshold schedule: `[50, 46, 43, 40, 37, 34, 31, 29, 27, 25, 23, 21, 19, 17, 15, 14, 13, 12, 11, 10]`

2000 particles, 20 generations

#### Result

![image-20200609181607116](../../../Library/Application Support/typora-user-images/image-20200609181607116.png)

​																												Figure 2

(Different colours indicates different generations)

![image-20200609181634239](../../../Library/Application Support/typora-user-images/image-20200609181634239.png)

​																												Figure 3

-   Greater M: lower acceptance rates at each generation, the more particles are needed in each generation, thus the more time is needed. 
-   Here, M=750 is the curve with lowest acceptance rates, as suggested by Filippi, S. et al, the maximal M=2000 (whole population) will be expected to have the lowest acceptance rates.
-   The traditional trivial multivariate normal kernel should have equal performance to M=2000, but here in `pyabc` the multivariate normal kernel is a modified version and thus better
-   Decrease the `scaling` argument of multivariate normal kernel will make the perturbation kernel more ‘local', as it is sampling under a ‘thinner’ Gaussian distribution, however there could be problems which are discussed in the next section

### Median eps with minimum eps=10

Using median epsilon schedule, starts from 50. Maximal generation numbers is set to 30, but ABC SMC will stop as long as the eps is smaller than 10

#### Result

![image-20200609190229514](../../../Library/Application Support/typora-user-images/image-20200609190229514.png)

​																												Figure 4

![image-20200609190324934](../../../Library/Application Support/typora-user-images/image-20200609190324934.png)

​																												Figure 5

-   New experiment added here: multivariate normal with grid search, provided py `pyabc`. It performs a grid search on the `scaling` argument of multivariate normal at each generation. In this case the `scaling` argument is changing among generations.
-   Similar result to the fixed threshold schedule. Multivariate normal with grid search does not give much improvement

### Trade-offs

As above figures suggest, the efficient settings are

-   Multivariate normal with M nearest neighbour, **with a small M**
-   Multivariate normal with a smaller `scaling` argument

They are considered because they take less sample numbers to reach a target threshold, i.e higher acceptance rates, fewer required samples, less execution time

However, they may have the drawbacks:

-   They may stuck in local optimum because they can hardly explore wider ranges (Figure 6). In this case, epsilon will not convergent to the zero; or they need much more samples to jump away from a local optimal (see ’NN M=50’ in Figure 2, were the last generation requires much more samples to find enough accepted particles). This situation could be worse if the target epsilon is set even smaller, e.g. 5. This agrees with conclusions from Filippi, S. et al 2013

-   Regardless of the above problem, if we only want a final population that targeting a given epsilon value, we could just 

    -   Firstly choose the kernels with the highest acceptance rates
    -   If two kernels have similar acceptance rates, choose the one with less computational complexity

    ![image-20200609200631811](../../../Library/Application Support/typora-user-images/image-20200609200631811.png)

    ​																												Figure 6

## Adaptive functions

>   Adapt the population size according to the mean coefficient of variation error criterion

-   Adaptive population: population at each generation varies from `min_population` to `max_population`
    -   `min_population=10`, `max_population=5000`
    -   Test result: after start, the population size is always adapted to `max_population=5000` in each of later generations, which results in a much longer execution time compared to constant population size = 2000
    -   Possible reasons
        -   Large number of parameters and large parameter space
    -   It might indicate that we should choose larger constant population size
    -   I’ll read the related paper (Klinger, 2017) and re-think about its appliance cases

## Adaptive distance

Efficiency in compared with 20 generations:

-   Number of required sample to reach eps_t
-   Acceptance rates and effective sampling size in each generations

Goodness of fit is observed from

-   Inferred parameter vs true parameter
-   Inferred curves vs observed data



## Stochastic acceptor



-   In future “Result Analysis” part, these experiment runs should be repeated for 5 or 10 times for a reliable result