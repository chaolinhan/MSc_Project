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