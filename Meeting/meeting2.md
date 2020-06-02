# Meeting 2

## 1. Tried to find an initial fit using 

-   AMC-SMC
    -   Not ideal
-   **Least square fit**

![image-20200531221119731](https://i.imgur.com/SjHvmwv.png)

```
{'iBM': 1.0267462374320455,
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
```

-   Increase data size: default 28 time points for N, $\Phi$, $\beta$ and $\alpha$, 112 data points in total, and they are easy to change

## 2. Write code 

-   Plot data
-   Python classes and functions

## 3. Read paper

To see how others do ABC-SMC, especially on how they set 

-   Distance function
-   Transition kernel 
-   Generation number and population size

And how to measure 

-   The algorithm efficiency 
-   The goodness of fit

## 4. Progress

1.  Distance function

    1.  I have changed the source code of distance function in `pyABC` to deal with missing values `NaN`. Now  the Euclidean distance function and adaptive function in `pyABC` supports data with `NaN`

    2.  I’m comparing the effect of adaptive functions

        (Left: adaptive, right: non-adaptive)

        ![image-20200531232459599](../../Library/Application Support/typora-user-images/image-20200531232459599.png)

2.  Before further experiments on the settings of ABC-SMC: how to measure 

    -   The algorithm efficiency:

        -   ESS
        -   Total sample size
        -   Acceptance rate
        -   (Execution time)

    -   The goodness of fit

        -   Mean and standard deviation of the final population
        -   Randomly sample K particles from the last population, plot the simulated data and mark the inter-quartile range (25% quantile and 75 quantile), e.g.
        
        -   Marginal posterior distribution for each of the 12 parameter, e.g.

![image-20200530181543558](https://i.imgur.com/45PuamA.png)
            
<img src="https://i.imgur.com/EQgyCrP.png" alt="image-20200530181600211" style="zoom: 50%;" />
            
<img src="https://i.imgur.com/F9sZFJQ.png" alt="image-20200530181707889" style="zoom:50%;" />
            
<img src="https://i.imgur.com/60KIn0w.png" alt="image-20200530181730819" style="zoom:50%;" />

3.  Add noise to the data:

    For N, $\Phi$, $\beta$, $\alpha$ data (artificial data):

    -   Measure their standard deviation $\sigma$ respectively: $\sigma_N$, $\sigma_\Phi$, $\sigma_\beta$ and $\sigma_\alpha$

    1.  Assumes the error term follows Gaussian distribution $N(0, \sigma^2)$
    2.  Sample error from the distribution: `sigma+np.random.randn(nr_sample)+0` for each simulated data point, using corresponding $\sigma$

## 5. Next steps in 2 weeks

Start experiments on:

-   Kernel: different kernel and theirs trade-offs
    -   Un iform
    -   Normal
    -   Multivariate normal
    -   Multivariate normal K neighbour (**MNN**)
    -   Multivariate normal OCM
-   Using adaptive distance function: apply more tweaking to see if it is worth in this problem
-   $\epsilon$ schedule, **adaptive population size**, prior range, 
-   Setup the environment in ARCHER and try some runs

## 6. Questions

-   Model misspecification?
    -   Current runs all result in a steady trend when time t >= 40 hr for all the four variables
    -   In real data, values still changes when  t >= 40
-   I have changed the source code of `pyABC`, which could be problematic when using ARCHER

# Meeting notes

-   Extend the time point to 120 hrs
    -   Now data size is 30*4=120 values 
-   For results: also plot the joint distribution of any two parameters, where we could possibly identify some patterns that some parameters are related and in a way we can reduce the number of parameters.
    -   We can identify some linear relations up to 2 or 3 orders e.g. x=ay^3+c, or exponential/log relations among two parameters but the joint parameter distribution cannot identify relations among three or more parameters
-   For question 6.1: one possible model is replace $\lambda_N$ by $\lambda_N e^{-at}$ which is an exponential decay term instead of a constant intercept
-   Sigma for error could be fixed, i.e. for N and Phi the error have the same sigma_1 and for beta and alpha the error have the same sigma_2. The sigma could be fixed, as is proposed above, according the variance go the raw data