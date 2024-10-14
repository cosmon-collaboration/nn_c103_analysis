We describe the features and usage of the software to analyze the NN correlators on the C103 ensemble.

The numerical correlators can be found at
- https://c51.lbl.gov/~walkloud/cosmon_corrs/

Note, this is an unprotected site (no username/passwd required).  We'll make the data public with arXiv anyways, so probably no one will find it before then.

For the code to run, we need to either rename these files or make soft links to them (after you download them) to match the convention in the input files,
```
mkdir data
cd data
wget https://c51.lbl.gov/~walkloud/cosmon_corrs/c103_nn_hdf5/c103_nn_hdf5/cls21_c103_r005-8_nucleon_S0.hdf5
...
ln -s cls21_c103_r005-8_nucleon_S0.hdf5 nucleon_S0.hdf5
...
```


The analysis proceeds in 3 steps:
- [single nucleon stability study](#single-nucleon)
- [two nucleon stability study](#two-nucleon)
- phase shift analysis

The analysis code requires a few Python libraries to work.  The version of each library with which this code has been tested are
- Python: 3.9.16
- numpy: 1.23.5
- opt_einsum: v3.3.0
- scipy: 1.10.0
- matplotlib: 3.6.3
- h5py: 3.7.0
- gvar: 11.11.1
- lsqfit: 13.0
! compatibility warning: lsqfit versions 13.2 and above encounter unresolved issues with the analysis.

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
The main code to run the fit is `nn_fit.py` and it expects an input file `nn_parameters.py`.  The two-nucleon fit can be performed to the two-nucleon correlators, or to the ratio of the two-nucleon to single nucleon correlators, by changing the flag in the input file
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
./plot_nn_stability_gevp.py result/NN_singlet_t0-td_4-10_N_n3_t_3-20_NN_conspire_e0_t_3-15_ratio_False.pickle
```
