# pysynthdid : Synthetic difference in differences for Python

## Reference: 
### original paper:
Arkhangelsky, Dmitry, et al. Synthetic difference in differences. No. w25532. National Bureau of Economic Research, 2019. https://www.nber.org/papers/w25532
### R pkg:
https://github.com/synth-inference/synthdid

## How to use:
See the jupyter notebook in [`notebook`](https://github.com/MasaAsami/pysynthdid/tree/main/notebook) for basic usage
- `ReproductionExperiment_CaliforniaSmoking.ipynb`
  - This is a reproduction experiment note of the original paper, using a famous dataset (CaliforniaSmoking).

- `OtherOmegaEstimationMethods.ipynb`
  - This note is a different take on the estimation method for parameter `omega` (& `zeta` ). As a result, it confirms the robustness of the estimation method in the original paper.

## Warning:
This module is still under development. Please check the logic carefully before using this. (Some optimization algorithms have been simplified.)

The following specifications will be added in the near future.
- Refactoring and better documentation
- Completion of the TEST code
- Calculation of SE code(Bootstrap Variance Estimation, Jackknife Variance Estimation)
- Enhanced visualization code
- Calculation of tau
