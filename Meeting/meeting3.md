# Measurement noise

![image-20200609170943091](../../../Library/Application Support/typora-user-images/image-20200609170943091.png)

-   (a): add noise to observed data and model (simulator)
-   (b): add noise to observed data
-   (c): no noise added



Adding noise makes the inference harder. Later implementations we could consider using 

-   A model with noise, or
-   [Stochastic acceptor](https://pyabc.readthedocs.io/en/latest/examples/noise.html) provided by `pyabc`

There are also papers using synthetic data with noise and model without noise to test the performance of ABC SMC

# Kernels

There is limited options in perturbation provided by `pyabc `. Other packages may provide more kernel options

Kernels provided by `pyabc`:

