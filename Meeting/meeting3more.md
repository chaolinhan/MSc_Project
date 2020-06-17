# Factor

<img src="../../../Library/Application Support/typora-user-images/image-20200612182157319.png" alt="image-20200612182157319" style="zoom:50%;" />

![image-20200612182215860](../../../Library/Application Support/typora-user-images/image-20200612182215860.png)

![image-20200612182347765](../../../Library/Application Support/typora-user-images/image-20200612182347765.png)

(Left to right: no factors applied, range factor and variance factor)

## Conclusion

Factors make ABC SMC more effective but less accurate in this case





# Less data, wider prior range

5000 particles, 30 generations

<img src="../../../Library/Application Support/typora-user-images/image-20200613200725117.png" alt="image-20200613200725117" style="zoom:50%;" />

![image-20200613200744799](../../../Library/Application Support/typora-user-images/image-20200613200744799.png)

![image-20200613200755704](../../../Library/Application Support/typora-user-images/image-20200613200755704.png)

![image-20200613201639735](../../../Library/Application Support/typora-user-images/image-20200613201639735.png)

(Left to right: basic run, less data, wider prior range)

## Conclusion

-   Using less data (40% of the original data size) does not reduce the required samples much
-   Using wider prior range will largely increase the required samples and takes much more time. After 30 generations the epsilon is still large
    -   For the real data inference, if we wish to set the prior range as small as possible. To do this, we can 
        -   Set the prior range of a parameter according to its biological meaning and try to make the range small
        -   Use a wide prior range (e.g. [0, 100]) for the first fit, set a smaller prior range according to the marginal distribution of the first fit, then do the second refined fit
            -   Might miss the true parameter



# Other findings

-   Many joint marginal distributions suggest a linear relationship between

    <img src="../../../Downloads/image-20200613213623271.png" alt="image-20200613213623271" style="zoom:20%;" />
    
-   $\lambda_\Phi$ and $\mu_\Phi$
    -   $\mu_\alpha$ and $s_{\alpha\Phi}$
    
    <img src="../../../Desktop/Screenshot 2020-06-14 at 23.28.32.png" alt="Screenshot 2020-06-14 at 23.28.32" style="zoom:50%;" />
    
    If later runs on the real data also shows these correlations, we might reduce these four parameters into two.





# Next steps

<img src="../../../Library/Application Support/typora-user-images/image-20200614235945503.png" alt="image-20200614235945503" style="zoom:50%;" />

## In the following three weeks

1.  Apply ABC SMC on the real data and try to obtain a small epsilon e.g. less than 5 at the end
2.  Model comparison
    -   Model 1: default ODEs
    -   Model 2: exponential decay $\lambda_N$: $\lambda_N\times \exp(-at)$
    -   Model 3: reduce the number of parameters if some parameters are highly correlated
3.  Possible: try ABC SMC on other packages if time allows
    -   Some other packages use different ways of implementations, the performance could be quite different

## After that, use another two weeks to 

1.  Do performance experiments on Cirrus/ARCHER

    Besides, try to profile the program, see if we could improve the performance

2.  If possible, accelerate the algorithm using GPU

## Final stage

1.  Do repeated experiments for more accurate results, and analysis the data
    -   Some other experiments could be done
        -   Compare to exact inference, traditional ABC rejection
2.  In the meantime, write the report and dissertation



# Meeting notes

-   Write the units of the four time series and try to limit their prior range
-   Set the covariance according to the real data at each time points
-   Try log uniform distribution in the prior distribution
-   New model: remove iBM
-   Try LS fitting with new models
-   Timer could be added to help profile





$d\beta/dt = AN-\mu_\beta\beta$

$\frac{s_{\beta N}}{1+i_{\beta\Phi}}=C$