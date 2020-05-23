# `pyABC`  Study Notes

[Paper Here](https://academic.oup.com/bioinformatics/article/34/20/3591/4995841)

- a scalable, dynamic parallelisation strategy

- adaptive population size selection

- adaptive, local transition kernels and acceptance threshold
schedules,
- configuration and extension without alterations of its source
code

Two scheduling options

- STAT static
- DYN dynamic

## Model

- Input: `dict`. Keys of the dict are the parameters of the model, in this case they are the parameters in the ODEs.
- Output: `dict`. In this case it is the simulated data from ODEs

### Notes on the runs

A drop in acceptance rate is usually used as termination criterion [more info needed]

Facing higher number of parameters, the number of particles in each population could be in=creased to avoid local minima

## Database

[Query the database](https://pyabc.readthedocs.io/en/latest/api_datastore.html#querying-the-database)

```python
df, w = history.get_distribution(m, t=19)
```
- df: DataFrame of particles in the population t
- w: weight of these particels

## Prior distribution

## Distance

### Static Euclidean distance that tolerates NAs(in `ODE.py`)

```python
def euclidean_distance(dataNormalised, simulation, normalise=True, time_length=9):
    if dataNormalised.__len__() != simulation.__len__():
        print("Input length Error")
        return

    if normalise:
        normalise_data(simulation)

    dis = 0.

    for key in dataNormalised:
        for i in range(time_length):
            if not np.isnan(dataNormalised[key][i]):
                dis += pow(dataNormalised[key][i] - simulation[key][i], 2.0)
            # NaN dealing: assume zero discrepancy
            else:
                dis += 0.
        # dis += np.absolute(pow((dataNormalised[key] - simulation[key]), 2.0)).sum()

    return np.sqrt(dis)
```

### Adaptive distance TBC

pyABC provide adaptive distance functions but compatibility with our ODEs should be studied, which are

- Multiple metrics of the data to be compared: N, M, A and B.
- Normalisation is used in static distance, in adaptive distance the solution should be taken care of

## A similar case

Several parameters and several target data objects (trajectories):
[Multi-scale model: Tumor spheroid growth](https://pyabc.readthedocs.io/en/latest/examples/multiscale_agent_based.html)

```python
from tumor2d import log_model, distance, load_default
from pyabc import ABCSMC
from pyabc.sampler import ConcurrentFutureSampler
from concurrent.futures import ThreadPoolExecutor

data_mean = load_default()[1]  # (raw, mean, var)

pool = ThreadPoolExecutor(max_workers=2)
sampler = ConcurrentFutureSampler(pool)

abc = ABCSMC(log_model, prior, distance,
             population_size=3,
             sampler=sampler)
```


## Web-based visualisation

### `bokeh` problem

```shell
 abc-server "test.db"
Traceback (most recent call last):
  File "/home/yuan/.local/bin/abc-server", line 5, in <module>
    from pyabc.visserver.server import run_app
  File "/home/yuan/.local/lib/python3.8/site-packages/pyabc/visserver/server.py", line 8, in <module>
    import bokeh.plotting.helpers as helpers
ModuleNotFoundError: No module named 'bokeh.plotting.helpers'
```

Reason: bokeh 2.0.0+ removed `plotting.helpers` entirely.

Solution: boken 1.4

Command: `abc-server /tmp/test.db`



## Weekly notes from 20 May

### Try more data points

-   Design
    -   Default
    -   `array([ 0.  ,  0.25,  0.5 ,  1.  ,  2.  ,  4.  ,  6.  ,  8.  , 10.  ,12.  , 14.  , 16.  , 18.  , 20.  , 22.  , 24.  , 28.  , 32.  , 36.  , 40.  , 44.  , 48.  , 52.  , 56.  , 60.  , 64.  , 68.  , 72.  ])`
    -   28 time points, 28*4=112 values in data

**DONE**

### Reset the initial parameters

-   Fit the data to get a least square parameters?
    -   Some lm fit not support multi variables LS
-   Turn to AMC-SMC for initial rough inference
    -   Initial population seems to take forever
    -   Code in `InitPara.py`
    -   The mean value of last population is quiet different from the peak value
        -   Use peak value?
        -   Use sum of value*weight **<- now chosen**

>   Issue spotted
>
>   -   Python 3.8.2 with pyABC will produce an error on the `model` parameter. Further attention should be paid.
>   -   Solution: use the old environment python 3.7
>
>   Bug spotted
>
>   -   Euclidean distance function return `NaN` in the test, which might bring the above problem “Initial population seems to take forever”
>   -   Solution: check the distance function. Unit test should also be modified to accommodated this

Result of `mean()`

```
iBM         9.737715
kMB        47.458632
kNB         9.973562
lambdaM    39.107247
lambdaN    25.262527
muA        47.041885
muB        15.762823
muM        39.539603
muN        79.994190
sAM        38.563204
sBN        37.618288
vNM        13.229111
```

Result of `df*w.sum()`:

```
iBM				9.051270
kMB       40.881926
kNB       9.618762
lambdaM   41.405661
lambdaN   29.360990
muA       44.426018
muB       16.450285
muM       37.356256
muN       78.150011
sAM       33.580249
sBN       41.486109
vNM       13.005909
```

![distribution](https://i.imgur.com/xihZ39S.png)