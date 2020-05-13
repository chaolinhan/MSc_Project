# pyABC Notes

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