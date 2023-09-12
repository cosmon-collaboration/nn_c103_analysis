We describe the features and usage of the software to analyze the NN correlators on the C103 ensemble.

The analysis proceeds in 3 steps:
- [single nucleon stability study](#single-nucleon)
- two nucleon stability study
- phase shift analysis

## Single Nucleon

We want to understand for a given `t_min` in the nucleon 2pt fit, what number of exponentials is optimal (or minimal) for obtaining a stable determination of the ground state nucleon energy.  The code to run the nucleon fit is
```
n_fit.py
```
which assumes an input file `n_parameters.py`.  There is a bash script, `nn_bash_scripts/run_nucleon.sh` that will create this file from `n_parameters_base.py`, changing the `t_min` and `n_states` in the analysis.