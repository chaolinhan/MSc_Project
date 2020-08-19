# Bayesian inference using sequential Monte-Carlo algorithms for dynamical systems models

## Project information

**Student name**: Chaolin Han

**Student number**: s1898201

**EPCC supervisor**: Mark Bull

**External supervisor**: Linus Schumacher (Centre for Regenerative Medicine, University of Edinburgh)

## Project description

Zebrafish can regenerate the spinal cord after injury, and possibly recover swimming function. The regeneration is a complex biological process with many interacting parts. We modelled the dynamics of two cell and two molecules that participate in the re- generation, according to the existing experimental findings[1], and wished to explore uncertainties in their parameter estimates.

As an Approximate Bayesian Computation (ABC) method, ABC SMC (Sequential Monte-Carlo) method was implemented on the proposed models to approximate the posteriors the parameters. We conducted some preliminary experiments to study the ability of inference and robustness of the algorithm before we performed the inference of parameters and model comparison using the same inference framework. The result shows well-inferred parameters and a preferred model with high model probability.

The algorithm implementation shows a decent scaling-up performance; however, prob- lems with local optimum and some other uncertainties were also found. Suggestions on the implementation options and ABC SMC settings were made to avoid these problems.

## Repository information

This git repository contains all source code used in the project, together with outputs and results of the executions, supervisor meeting minutes, input data, etc. The directory structure and descriptions are listed below.

```
.
├── README.md
├── code
│   ├── TODO
│   ├── data/
│   │   └── TODO
│   ├── db/ -> TODO
│   ├── log_uniform/
│   │   └── TODO
│   ├── performance/
│   │   └── TODO
│   ├── pyABC_study/
│   │   └── TODO
│   └── uniform/
│       └── TODO
├── meeting/
│   └── TODO
├── requirements.txt
│   └── TODO
└── results/
    └── TODO
```




## Reference 

> 1. T. M. Tsarouchas, D. Wehner, L. Cavone, T. Munir, M. Keatinge, M. Lambertus, A. Underhill, T. Barrett, E. Kassapis, N. Ogryzko, et al., “Dynamic control of proinflammatory cytokines il-1β and tnf-α by macrophages in zebrafish spinal cord regeneration,” Nature communications, vol. 9, no. 1, pp. 1–17, 2018.
