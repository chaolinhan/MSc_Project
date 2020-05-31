# ABC-SysBio

-   Considerable care must be taken to choose appropriate, informative summaries of the data 29-33
    -   In the context of **model selection**, the use of summarised data is particularly problematic
-   Many of the settings will affect convergence to the true parameter
    -   The number of particles must be large enough to cover the entire parameter search space
        -   Increase with the number of parameters
    -   Epsilon: quantile epsilon
-   Prior distribution
    -   Normal distribution
    -   Log normal
    -   Uniform <-
    -   Log uniform <-
    -   Constant
-   Distance 
    -   Sum of square errors (Euclidean distance)
    -   Adaptive distance function
    -   Noise model
-   Kernel 
    -   Uniform
    -   Normal
    -   Multivariate normal
    -   Multivariate normal K neighbour
    -   Multivariate normal OCM

# Optimizing Threshold–Schedules

-   Showed: the current preferred method of choosing thresholds as a pre-determined quantile of the distances between simulated and observed data from the previous population, can lead to the inferred posterior distribution **being very different to the true posterior**.
-   Thus threshold `eps` selection is an important challenge
-   Proposed a better eps schedule

# Approximate Bayesian Computation for infectious disease modelling

Offers three case study about ABC implementation 

## About kernels

A perturbation kernel with a wide variance will stop the algorithm from being stuck in local modes, but will lead to a large number of particles being rejected, and thus cause the algorithm to be inefficient. 

Mentioned: **multivariate normal** and **MNN**

For MNN, normalised Euclidean distance can be used.

## Case 1

-   No noise aplplied
-   Using full data is better than using summary statistics of the data
-   Effect of different data size

## Case 2

-   Behaviour of **ESS**

-   ABC-SMC is much effective than ABC rejection by saving sampling size, under a fixed schedule of $\epsilon$

>   Convergence can be assessed by the difference using the inter-quartile ranges of the values of accepted particles as a measure of goodness of fit between successive intermediate distributions (Toni et al., 2009).

# On optimality of kernels for approximate Bayesian computation using sequential Monte Carlo

Computational efficiency for different kernels