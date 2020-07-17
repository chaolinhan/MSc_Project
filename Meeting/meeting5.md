# Extra terms for tnf-α

1.  Original (model 3)

$$
\frac{\mathrm{d} \alpha}{\mathrm{d} t}=s_{\alpha\Phi}\Phi-\mu_\alpha\alpha
$$

2.  Model 4

$$
\frac{\mathrm{d} \alpha}{\mathrm{d} t}=s_{\alpha\Phi}\Phi-\mu_\alpha\alpha+d_{\beta\alpha}\beta
$$

3.  Model 5

$$
\frac{\mathrm{d} \alpha}{\mathrm{d} t}=(s_{\alpha\Phi}+f_{\beta\alpha}\beta)\Phi-\mu_\alpha\alpha
$$

# Other experiments

-   Factors
    
-   Tested on model 4
    
-   Population size
    -   Adaptive distance in `pyabc` have a bug, for now I cannot apply it on the observed data. I tried to modify the source code but it does not work as the bug involves may other packages
    -   So I turned to fixed population size 5,000 and 10,000

-   I checked the raw data distribution. The most abnormal distribution is from hpl 120 for macrophage

    ![image-20200628224617798](../../../Library/Application Support/typora-user-images/image-20200628224617798.png)

    

    