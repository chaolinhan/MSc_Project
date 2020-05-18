# 1

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

    **1 - 2 weeks**

2.  Use the true experiment data and perform ABC-SMC

    **Followed weeks**