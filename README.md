# Bayesian inference using sequential Monte-Carlo algorithms for dynamical systems models

## Project information

**Student name**: Chaolin Han

**Student number**: s1898201

**EPCC supervisor**: Mark Bull

**External supervisor**: Linus Schumacher (Centre for Regenerative Medicine, University of Edinburgh)

## Project abstract

Zebrafish can regenerate the spinal cord after injury, and possibly recover swimming function. The regeneration is a complex biological process with many interacting parts. We modelled the dynamics of two cell and two molecules that participate in the re- generation, according to the existing experimental findings[[1](###reference)], and wished to explore uncertainties in their parameter estimates.

As an Approximate Bayesian Computation (ABC) method, ABC SMC (Sequential Monte-Carlo) method was implemented on the proposed models to approximate the posteriors the parameters. We conducted some preliminary experiments to study the ability of inference and robustness of the algorithm before we performed the inference of parameters and model comparison using the same inference framework. The result shows well-inferred parameters and a preferred model with high model probability.

The algorithm implementation shows a decent scaling-up performance; however, prob- lems with local optimum and some other uncertainties were also found. Suggestions on the implementation options and ABC SMC settings were made to avoid these problems.

## Repository information

This git repository contains all source code used in the project, together with outputs and results of the executions, supervisor meeting minutes, input data, etc. The directory structure and descriptions are listed below.

```shell
.
├── README.md
├── code/
│   ├── data/
│   ├── db/
│   ├── log_uniform/
│   ├── performance/
│   ├── pyABC_study/
│   └── uniform/
├── meeting/
├── requirements.txt
└── results/
```

- `requirements.txt` is the Python package version requirements
- `code/` contains all source code and outputs
  - `data/`: input data
  - `db/`: path for the database files. Output database files are not include as they are too large
  - `log_uniform/`: code files for the log-uniform prior experiments
  - `uniform/`: code files for the uniform prior experiments
  - `performance/`: code files for performance experiments
  - `pyABC_study/`: code for infer-back experiments, ODEs, plots and results analysis
- `meeting/` contains meeting minutes with supervisors 
- `results/` contains some tables and figures for results analysis

## Run the code

To run the Python code and replicate the experiments

- install required packages in `requirements.txt`
- ensure Python multi-core processing is enabled
- change the I/O path and the algorithm settings in the code
- for implementation code: `python3 <filename>.py > output.txt`
- for analysis code: run cell blocks in Python console

To run the code on machines that support `PBS` or `Slurm`

- change the task name and required resources in the provided `pbs` or `slurm` file
- submit the job using 
  - `qsub <filename>.pbs`, or
  - `sbatch <filename>.slurm`

## Input and output

The input file is `code/data/rawData.csv`. Some other input data are integrated in `code/pyABC_study/ODE.py`.

The output includes a database file ended with `.db` which contains information about each generation and algorithm setting. To process and analysis the `db` file, see [here](https://pyabc.readthedocs.io/en/latest/api_datastore.html). The analysis code read directly form the `db` file.

## Environment

The implementation run smoothly in the following environment.

- Cirrus[[2](###reference)]
  - Operating system: Red Hat Enterprise Linux 8.1
  - miniconda environment: 4.8.3
  - Python interpreter: 3.7.7
  - gcc: 6.3.0
- Local laptop
  - Operating system: macOS Catalina 10.15.6
  - PyCharm (IDE): Professional 2020.2
  - Python interpretor: 3.7.0
  - IPython: 7.12.0
  - Clang: 6.0 (clang-600.0.57)

The analysis code are tested on local machine.

## Reference 

> 1. T. M. Tsarouchas, D. Wehner, L. Cavone, T. Munir, M. Keatinge, M. Lambertus, A. Underhill, T. Barrett, E. Kassapis, N. Ogryzko, et al., “Dynamic control of proinflammatory cytokines il-1β and tnf-α by macrophages in zebrafish spinal cord regeneration,” Nature communications, vol. 9, no. 1, pp. 1–17, 2018.
> 2. https://www.cirrus.ac.uk/about/hardware.html
