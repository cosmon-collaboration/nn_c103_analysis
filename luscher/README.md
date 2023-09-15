We describe the features and usage of the software to analyze the NN correlators on the C103 ensemble.

The numerical correlators can be found at
- https://c51.lbl.gov/~walkloud/cosmon_corrs/

Note, this is an unprotected site (no username/passwd required).  We'll make the data public with arXiv anyways, so probably no one will find it before then.

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

After running this script, we can run
```
python plot_nucleon_stability.py
```
which will generate stability plots for the single nucleon fits with the various momentum boosts that we are interested it for this project: `[0, 1, 2, 3, 4, 5F1, 5F2]`