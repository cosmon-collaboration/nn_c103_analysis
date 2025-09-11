We describe the features and usage of the software to analyze the NN correlators on the C103 ensemble.

The numerical correlators can be found at
- https://portal.nersc.gov/cfs/m2986/cosmon/nn_c103_2505.05547/

The data files used in the Luscher analysis are
- cosmon_c103_r005-8_nucleon.hdf5
- cosmon_c103_r005-8_deuteron_Swave.hdf5
- cosmon_c103_r005-8_dineutron_Swave.hdf5

Download these files and place them in a data folder, eg.
```
mkdir data
wget wget -nd -r -P data -A "cosmon_c103_r005-8_*.hdf5" https://portal.nersc.gov/cfs/m2986/cosmon/nn_c103_2505.05547/
```

In order to reproduce the main result from the paper, the `nn_parameters.py` file is provided with the repo.  This is the input file for `nn_fit.py`.  With the data files downloaded, the fit should work with by running `./nn_fit.py`

NOTE: there is a bug with this code and lsqfit >= 13.1 that is not yet resolved.  In order to run the fit, an environment with the following libraries should work
- Python: 3.11.16
- numpy: 1.26.4
- opt_einsum: v3.3.0
- scipy: 1.11.3
- matplotlib: 3.8.0
- h5py: 3.8.0
- gvar: 11.11.15
- lsqfit: 13.0.1

The input file is mostly self-descriptive with some idiosyncrasies needed for it to work.  Comments in the input file are hopefully sufficiently descriptive to guide usage.

There are helper scripts in the `nn_bash_scripts` folder that loop over many fitting choices discussed in the paper:
- GEVP times `t0` and `td`
- number of single nucleon states in fit model
- whether fit model is `conspiracy` or `agnostic`
- `t_min` of the single nucleon and NN correlators
- `t_max` of the NN correlators

In general, the analysis proceeds in 3 steps:
- [single nucleon stability study](#single-nucleon)
- [two nucleon stability study](#two-nucleon)
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

## Two Nucleon

For the two-nucleon fits, we have two models to try, `conspire` and `agnostic`.
The main code to run the fit is `nn_fit.py` and it expects an input file `nn_parameters.py`.  The version of `nn_parameters.py` in the repo is the chosen "final" fit.  

If the data files `cosmon_c103_r005-8_*` are placed in a `data` folder, and the fit is run
```
./nn_fit.py
```
it should run and produce results from the paper.  To get the bootstrap samples, the flag
```
p["bootstrap"] = False
```
should be changed to `True`.  Of note, one FIRST has to run the non-bootstrap fit before running the bootstrap fits.

The two-nucleon fit can be performed to the two-nucleon correlators, or to the ratio of the two-nucleon to single nucleon correlators, by changing the flag in the input file
```
    p["ratio"]      = False # fit NN and N1 and N2 
    p["ratio"]      = True  # fit NN/N1/N2 and N1 and N2
```
The script `nn_bash_scripts/run_nn_agnostic_noRatio.sh` will loop over various choices of `t_min` for the nucleon as well as two-nucleon correlator.  It will also loop over the number of states used for the nucleon and two-nucleon.  Importantly, one has to chose the value of `t0` and `td` for the GEVP, this is not looped over.  In order to create the plots for the paper, we need to run with the `t0-td` values of 
- 3-8
- 3-10
- 4-8
- 4-10
- 5-10
- 6-10

Given these values, the stability plots versus GEVP times is obtained with `plot_nn_stability_gevp.py`.  This script requires an `optimal` fit, chosen by the user, from the various results obtained.  For example:
```
./plot_nn_stability_gevp.py result/NN_dineutron_tnorm3_t0-td_5-10_N_n3_t_4-20_NN_conspire_e0_t_4-15_ratio_False_block8.pickle
```
The other plotting routines to create stability plots function similarly.

## phase shift analysis

Given a set of bootstrap results from the spectrum, one can run the phase shift analysis, for example
```
./qcotd_inverse_qsq.py result/NN_deuteron_tnorm3_t0-td_5-10_N_n3_t_4-20_NN_conspire_e0_t_4-15_ratio_False_block8.pickle
```
There are a few options for controlling the analysis which are described with the help option, `-h`.  Important in this analysis was the use of `--irrep_avg` for the deuteron and using the continuum dispersion relation, `--continuum_disp` when implementing the quantization condition.
