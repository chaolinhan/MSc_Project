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

# Toni, Welch, Strelkowa et al, 2009

-   Bayes factor in model selection

## LV model example

-   Gaussian noise N(0, 0.25)
-   Euclidean distance
-   Tolerance $\epsilon$ chosen according to the artificially added noise

## Repressilator model

Add code to display the inter-quartile range vs generations plot

-   Meaning: the narrower the interval for a given tolerance et, the more sensitive the model is to the corresponding parameter.

**Benefit of ABC SMC:** sencitivity

>   ABC SMC recovers the intricate link between model sensitivity to parameter changes and inferability of parameters. … ABC SMC provides us with a global parameter sensitivity analysis (Sanchez & Blower 1997) on the fly as the intermediate distributions are being constructed

PCA and sensitivity:

>   In contrast to the interest in the first PC in most PCA applications, our main interest lies in **the smallest PC**. The last PC extends across the narrowest region of the posterior parameter distribution, and therefore provides information on parameters to which the model is the most sensitive. In other words, the smaller PCs correspond to stiff parameter combinations, while the larger PCs may correspond to sloppy parameter combinations (Gutenkunst et al. 2007).

Sensitivity:

>   Analysing and comparing the results of the deterministic and stochastic repressilator dynamics shows that parameter sensitivity is intimately linked to inferability. If the system is insensitive to a parameter, then this parameter will be hard (or even impossible) to infer, as varying such a parameter does not vary the output—which here is the approximate posterior probability—very much. In stochastic problems, we may furthermore have the scenario where the fluctuations due to small variations in one parameter overwhelm the signals from other parameters.

Noise and model selection: 

>   If the data are very noisy (Gaussian noise with standard deviation sZ1 was added to the simulated data points), then the algorithm cannot detect a single best model, which is not surprising given the high similarity of model outputs.

![image-20200602200204326](https://i.imgur.com/MjdodYN.png)

# Approximate Bayesian computation applied to the identification of thermal damage of biological tissues due to laser irradiation

-   Sensitivity visualisation
-   Estimation is far from the true one
-   Small prior lead to correct true data