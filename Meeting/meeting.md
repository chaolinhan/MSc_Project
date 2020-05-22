# Meeting 1

-   Time: 1500 18 May
-   Present: all

## Prior distribution

Initialise the prior distribution of parameters according to their real values.

-   Uniform distribution

-   1000 particles in each population

-   Use a small interval e.g. [-1. 1] for parameters:

    -   Eps -> 12

    Results:

    ![image-20200518140955266](https://i.imgur.com/a1Jzj9P.png)

-   Use a large interval e.g. [-50, 50] for parameter:

    -   Eps -> 6
    -   Lead to other optimal that is much different from the true parameter

-   Expected final eps show close to zero but now these runs all end up with a eps value >= 5.00





## Next steps

![image-20200518142624504](https://i.imgur.com/LAOSEXu.png)

1.  Use artificial data with known true parameters:

    1.  Study the performance and quality of inference of different hyper parameters and raw data size
    2.  Study different adaptive distance and transition kernels in ABC-SMC in pyABC
    3.  I found several similar examples of ABC-SMC in some papers and software documentations. I could look into them to see their setting of ABS-SMC and analysis of results
    4.  I also found examples of exact inference. I’ll try it and compare it will ABC

    **5 weeks** estimated

2.  Use the true experiment data and perform ABC-SMC

    **Followed weeks**

## Meeting notes

-   Prior distribution
    -   Under current ODEs, the 12 parameters should be positive
    -   The lower bound should be 0, the upper bound can be tried from 1 to 100 for each 
    -   The scales of the parameters should be justified according to the significance in the dynamical system and the data
    -   Distribution type limited to uniform and log uniform now
-   The data size should be increased, i.e. more time points
-   Distance function
    -   Check the normalisation. If necessary, then we could look at different algorithm of normalisation and  how they affect the result
    -   Try adaptive distance function
-   Also adaptive  population
-   Read some paper to find
    -   How the result is evaluated
    -   How to do exact inference (assume Gaussian distribution) and compare it to ABC-SMC
    -   How do they say up the experiments
-   Try fewer population but with more particles 