# pysynthdid : Synthetic difference in differences for Python

## Original Paper: 
Arkhangelsky, Dmitry, et al. Synthetic difference in differences. No. w25532. National Bureau of Economic Research, 2019. https://www.nber.org/papers/w25532

## How to use:
See the jupyter notebook in `notebook` for basic usage
- `ReproductionExperiment_CaliforniaSmoking.ipynb`
  - This is a reproduction experiment note of the original paper, using a famous dataset (CaliforniaSmoking).

- `OtherOmegaEstimationMethods.ipynb`
  - This note examines CrossValidation for `zeta` and constraint relaxation for `omega`.


## Warning:
This module is still under development. Please check the logic carefully before using this. (Some optimization algorithms have been simplified.)

The following specifications will be added in the near future.
- Refactoring and better documentation
- Completion of the TEST code
- Calculation of SE code(Bootstrap Variance Estimation, Jackknife Variance Estimation)
- Enhanced visualization code
- Calculation of tau
