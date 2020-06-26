# Measure noise and the raw data

1.  The raw observed at each time points shows a high variance, the real trend of these four times series might differ from the current one

    e.g. the measured number of macrophages at time point hpl 120![image-20200621145806222](../../../Library/Application Support/typora-user-images/image-20200621145806222.png)

2.  The see how small the $\epsilon_t$ we want, I added a Gaussian distributed noise to the time series of N and $\Phi$. The raw data of $\beta$ and $\alpha$ just give the error of the mean, I did not know the measurement population size so I cannot get the standard deviation, thus I did not add noise to these two time series

    The distance between noise data and observed data is 19.21 (average of 10)

# Units of parameters

Suppose in the equation and observed data, $N$ and $\Phi$ are count as ‘cell’ and the levels of il-1β and tnf-α expression are count as ‘unit’ (relative to t=0) and time unit is h (hour), then the units of these parameters is

![image-20200621154622398](../../../Library/Application Support/typora-user-images/image-20200621154622398.png)

-   QUESTION: $k_{N\beta}$ and $k_{\Phi\beta}$ is of the same unit. Is it possible that $k_{N\beta}$ is 70 times bigger than $k_{\Phi\beta}$ (which is observed in a result of AMC SMC run)?



# Goodness of fit

The results in `abcsmc1.pdf` shows that our best model so far can fit the curve of $N$ and $\beta$ generally well, but for $\Phi$ and $\alpha$ is less satisfactory

![image-20200621160613684](../../../Library/Application Support/typora-user-images/image-20200621160613684.png)

-   QUESTION do we need some extra term?

    -   e.g. in the original equation
        $$
        \frac{\mathrm{d} \alpha}{\mathrm{d} t}=s_{\alpha\Phi}\Phi-\mu_\alpha\alpha
        $$
        Levels of α is promoted by $\Phi$ and inhibited by $\alpha$ itself

        Is combination of the two effects insufficient to produce a curve as the observed data?



# Next steps

1.  Wait for the results of prior range [0, 50]
2.  Try: larger populations, factors, adaptive functions and `LocalTransition()`
3.  Get prepared for performance experiments



# Meeting notes

-   Experiments 
    -   Applying factors s.t. data at the front more important
    -    Using adaptive distance function, allowed population up to 10,000, allowed variance percent up to 10%
-   Plot the raw data error, check if the raw data is of Gaussian distribution

