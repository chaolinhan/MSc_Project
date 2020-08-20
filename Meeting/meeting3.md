# Meeting 3

-   Time: 3pm, 15 June 
-   Present: all

# Things to report

## Measurement noise

![image-20200609170943091](../../../Library/Application Support/typora-user-images/image-20200609170943091.png)

​																												Figure 1

-   (a): add noise to observed data and model (simulator)
-   (b): add noise to observed data
-   (c): no noise added



Adding noise makes the inference harder. Later implementations we could consider using 

-   A model with noise, or
-   [Stochastic acceptor](#Stochastic-acceptor)

There are also papers using synthetic data with noise and model without noise to test the performance of ABC SMC

## Kernels - efficiency of algorithm

Goal: find the most efficient kernels under a given schedule.

Give a target threshold eps_t, the efficiency can be measure by

1.  Number of required sample to reach eps_t
2.  Acceptance rates and effective sampling size in each generations

Related paper:

>   Filippi, S., Barnes, C. P., Cornebise, J., & Stumpf, M. P. H. (2013). On optimality of kernels for approximate Bayesian computation using sequential Monte Carlo. *Statistical Applications in Genetics and Molecular Biology*, *12*(1), 87–107. https://doi.org/10.1515/sagmb-2012-0069

There is limited options in perturbation kernels provided by `pyabc `. Other packages may provide more kernel options.

Kernels provided by `pyabc`:

-   Multivariate normal kernel
    -   One parameter: `scaling` default to 1
    -   Scaling is a factor which additionally multiplies the covariance with
-   Multivariate normal kernel with M nearest neighbours 
    -   For each particle, M-nearest neighbours are selected, the covariance is calculated using these neighbours 

### Experiments

In future “Result Analysis” part, these experiment runs should be repeated for 5 or 10 times for a reliable result

#### Fixed threshold schedule

Under a fixed threshold schedule: `[50, 46, 43, 40, 37, 34, 31, 29, 27, 25, 23, 21, 19, 17, 15, 14, 13, 12, 11, 10]`

2000 particles, 20 generations

#### Result

![image-20200609181607116](../../../Library/Application Support/typora-user-images/image-20200609181607116.png)

​																												Figure 2

(Different colours represents different generations)

![image-20200609181634239](../../../Library/Application Support/typora-user-images/image-20200609181634239.png)

​																												Figure 3

-   Greater M in Nearest Neighbour (NN): lower acceptance rates at each generation, the more particles are needed in each generation, thus the more time is needed. 
-   Here, M=750 is the curve with lowest acceptance rates, as suggested by Filippi, S. et al, the maximal M=2000 (whole population) will be expected to have the lowest acceptance rates among all M.
-   The traditional trivial multivariate normal kernel should have equal performance to M=2000, but here in `pyabc` the multivariate normal kernel is a modified version and thus better
-   Decrease the `scaling` argument of multivariate normal kernel will make the perturbation kernel more ‘local', as it is sampling under a ‘thinner’ Gaussian distribution, however there could be problems which are discussed in the next section

#### Median threshold schedule with minimum eps=10

Using median epsilon schedule starting from 50. Maximal generation numbers is set to 30, but ABC SMC will stop as long as the eps is smaller than 10

#### Result

![image-20200609190229514](../../../Library/Application Support/typora-user-images/image-20200609190229514.png)

​																												Figure 4

![image-20200609190324934](../../../Library/Application Support/typora-user-images/image-20200609190324934.png)

​																												Figure 5

-   New experiment added here: multivariate normal with grid search, provided py `pyabc`. It performs a grid search on the `scaling` argument of multivariate normal at each generation. In this case the `scaling` argument is changing among generations
-   Similar result to the fixed threshold schedule. Multivariate normal with grid search does not give much improvement

### Trade-offs

As above figures suggest, the efficient settings are

-   Multivariate normal with M nearest neighbour, **with a small M**
-   Multivariate normal with a **smaller `scaling` argument**

They are considered because they take less sample numbers to reach a target threshold, i.e higher acceptance rates, fewer required samples, less execution time.

However, they may have drawbacks:

-   They may stuck in local optimum because they can hardly explore wider ranges (Figure 6). In this case, epsilon will not convergent to zero; or they need much more samples to jump away from a local optimal (see ’NN M=50’ in Figure 2, where the last generation requires much more samples to find enough accepted particles). This situation could be worse if the target epsilon is set even smaller, e.g. 5. This agrees with conclusions from Filippi, S. et al 2013

-   Regardless of the above problem, if we only want a final population that targets a given epsilon value, we could just 

    -   Firstly choose the kernels with the highest acceptance rates
    -   If two kernels have similar acceptance rates, choose the one with less computational complexity

    ![image-20200609200631811](../../../Library/Application Support/typora-user-images/image-20200609200631811.png)

    ​																												Figure 6

### Conclusion

On a given final eps, using small `scaling` in multivariate normal and NN with small M is efficient, but if we are targeting the true posterior we should consider original multivariate normal or NN with M>=100





## Adaptive functions

### Adaptive population

>    Adapt the population size according to the mean coefficient of variation error criterion

-   Adaptive population: population at each generation varies from `min_population` to `max_population`
    -   `min_population=10`, `max_population=5000`
    -   **Test result**: after start, the population size is always adapted to `max_population=5000` in each of later generations, which results in a much longer execution time compared to constant population size = 2000
    -   Possible reasons
        -   Large number of parameters and large parameter space
    -   It might indicate that we should choose larger constant population size
    -   I’ll read the related paper (Klinger, 2017) and re-think about its appliance cases
    
    

### Adaptive distance

Efficiency is compared with 20 generations':

-   Number of required sample to reach eps_t
-   Acceptance rates and effective sampling size in each generations

Goodness of fit is observed from

-   Inferred parameter vs true parameter
-   Inferred curves vs observed data

#### Distance function: assume Euclidean distance

$D=(\Sigma_i (w_if_i\Delta x_i)^2)^{1/2}$

$\Delta x=x_{simulated}-x_{observed}$, w is **weight**, f is **factor**

-   Non-adaptive distance: w and f is always 1

-   Adaptive distance: $w_i$ is changing among generations, in order to give informative data points higher weight
-   Factors could be given as an normalisation. When some data points are equally informative, applying factors could take their scales into account
    -   In this experiment, factors are set proportional to the data ranges of the four curves
    -   More factors could be tried later

### Results

#### Efficiency

![image-20200610160513484](../../../Library/Application Support/typora-user-images/image-20200610160513484.png)

​																												Figure 7

![image-20200610160527767](../../../Library/Application Support/typora-user-images/image-20200610160527767.png)

​																												Figure 8

Running 20 generations:

-   Adding factors make the non-adaptive distance more efficient
-   Adaptive distance with/without factors applied seems to be much more efficient, but they have accuracy problem (below)

### Goodness of fit

-   The distance threshold i.e. eps is not comparable between adaptive ones and non adaptive ones because of different weights

![image-20200610161423332](../../../Library/Application Support/typora-user-images/image-20200610161423332.png)

​																												Figure 9

(From left to right: non-adaptive distance, non-adaptive distance with factors, adaptive distance, adaptive distance with factors)

-   Adaptive distance might just not suitable for this task
    -   Too many data points, tool large parameter space will probably make the ‘adaptive’ distance function hard to identify which data points are more informative in inferring than others

### Conclusion

-   Factor would help with non-adaptive distance function, adaptive distance function might be not helpful in this case



## Stochastic acceptor 

In `pyabc`, measurement noise can be consider via 

1.  Manually add a noise term to the model, or
2.  Using a stochastic acceptor

If we use a stochastic acceptor, then 

-   Distance function should be changed in order to consider the noise term
-   Epsilon schedule switched to a [temperature schedule](https://pyabc.readthedocs.io/en/develop/examples/noise.html)

Tests on my local laptop shows bad results, Experiments with larger problem size on ARCHER are still **in progress**



## Planned experiments

The above experiments is mostly about the efficiency, the following experiments will study more on the goodness of fit:

-   Prior range
-   Data size
-   Population size and number of populations

However these experiments are less important, as for the real data from Tsarouchas et al., we can only start trying with wide prior range, use all the data points available, and run ABC SMC with large population and more generations.



## Factor

<img src="../../../Library/Application Support/typora-user-images/image-20200612182157319.png" alt="image-20200612182157319" style="zoom:50%;" />

![image-20200612182215860](../../../Library/Application Support/typora-user-images/image-20200612182215860.png)

![image-20200612182347765](../../../Library/Application Support/typora-user-images/image-20200612182347765.png)

(Left to right: no factors applied, range factor and variance factor)

### Conclusion

Factors make ABC SMC more effective but less accurate in this case





## Less data, wider prior range

5000 particles, 30 generations

<img src="../../../Library/Application Support/typora-user-images/image-20200613200725117.png" alt="image-20200613200725117" style="zoom:50%;" />

![image-20200613200744799](../../../Library/Application Support/typora-user-images/image-20200613200744799.png)

![image-20200613200755704](../../../Library/Application Support/typora-user-images/image-20200613200755704.png)

![image-20200613201639735](../../../Library/Application Support/typora-user-images/image-20200613201639735.png)

(Left to right: basic run, less data, wider prior range)

### Conclusion

-   Using less data (40% of the original data size) does not reduce the required samples much
-   Using wider prior range will largely increase the required samples and takes much more time. After 30 generations the epsilon is still large
    -   For the real data inference, if we wish to set the prior range as small as possible. To do this, we can 
        -   Set the prior range of a parameter according to its biological meaning and try to make the range small
        -   Use a wide prior range (e.g. [0, 100]) for the first fit, set a smaller prior range according to the marginal distribution of the first fit, then do the second refined fit
            -   Might miss the true parameter



## Other findings

-   Many joint marginal distributions suggest a linear relationship between

    <img src="../../../Downloads/image-20200613213623271.png" alt="image-20200613213623271" style="zoom:20%;" />

-   $\lambda_\Phi$ and $\mu_\Phi$

    -   $\mu_\alpha$ and $s_{\alpha\Phi}$

    <img src="../../../Desktop/Screenshot 2020-06-14 at 23.28.32.png" alt="Screenshot 2020-06-14 at 23.28.32" style="zoom:50%;" />

    If later runs on the real data also shows these correlations, we might reduce these four parameters into two.





# Next steps

<img src="../../../Library/Application Support/typora-user-images/image-20200614235945503.png" alt="image-20200614235945503" style="zoom:50%;" />

## In the following three weeks

1.  Apply ABC SMC on the real data and try to obtain a small epsilon e.g. less than 5 at the end
2.  Model comparison
    -   Model 1: default ODEs
    -   Model 2: exponential decay $\lambda_N$: $\lambda_N\times \exp(-at)$
    -   Model 3: reduce the number of parameters if some parameters are highly correlated
3.  Possible: try ABC SMC on other packages if time allows
    -   Some other packages use different ways of implementations, the performance could be quite different

## After that, use another two weeks to 

1.  Do performance experiments on Cirrus/ARCHER

    Besides, try to profile the program, see if we could improve the performance

2.  If possible, accelerate the algorithm using GPU

## Final stage

1.  Do repeated experiments for more accurate results, and analysis the data
    -   Some other experiments could be done
        -   Compare to exact inference, traditional ABC rejection
2.  In the meantime, write the report and dissertation



# Meeting notes and actions

-   Write the units of the four time series and try to limit their prior range
-   Set the covariance according to the real data at each time points
-   Try log uniform distribution in the prior distribution
-   New model: remove iBM
-   Try LS fitting with new models
-   Timer could be added to help profile

