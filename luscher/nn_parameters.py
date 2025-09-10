import numpy as np

def params():
    """
    User input file to control fitting options.

   """
    import numpy as np

    p = dict()
    p["debug"]   = False #True to turn on some debugging messages
    p["verbose"] = False #True to turn on some messages
    p["latex"]   = True #True to use latex with matplotlib

    # path to data files
    p["fpath"] = {"nucleon": "./data/cosmon_c103_r005-8_nucleon.hdf5", 
                  "nn": "./data/cosmon_c103_r005-8_deuteron_Swave.hdf5"}
    # save the fit results?
    p["save"] = True
    # what minimizer to use, 'scipy_least_squares', 'gsl', ...
    p["fitter"] = 'scipy_least_squares'

    p["t0"] = 5  # t0 for GEVP rotation matrix
    p["td"] = 10 # td for GEVP rotation matrix
    p['t_norm'] = 3 # time to normalize correlators
    p['gevp']   = 'evp' # evp or gevp
    p['get_Zj'] = True # compute overlap of each operator onto each state
    # where to save the overlap factors
    if 'deuteron' in p["fpath"]["nn"]:
        p['Zjn_values'] = f"result/deuteron_Zjn_tNorm{p['t_norm']}_{p['gevp']}.h5"
    elif 'dineutron' in p["fpath"]["nn"]:
        p['Zjn_values'] = f"result/dineutron_Zjn_tNorm{p['t_norm']}_{p['gevp']}.h5"
    p['show_Zjn']   = False # plot Zij values
    p['do_gevp']    = False #set to True if you want to do gevp even if it was already done and saved

    p["block"] = 8 # how many neighboring configurations to average together
    #p['cfgs']  = [0,802[,1]] # use this to cut configs if desired 

    p['svd_study'] = False #perform an svd study when fitting correlators
    p['svdcut']    = 1e-8 # svd cut to use

    p["bootstrap"] = False # one must run w/out BS first (False) before running the BS samples
    p['Nbs_max']   = 5000 # max amount of BS samples
    p['bs_seed']   = 'nn_c103_b%d' %p["block"] # seed the random number generator
    p["nbs"]       = 5000 # how many BS samples to do
    p["nbs_sub"]   = 100 # how many BS samples to do before saving results
    p['bs0_width'] = 5 # during BS resampling, set the width of priors to 5*sigma where sigma is the gs uncertainty determined with boot0.  This is to stabilize BS resampling and help ensure the BS samples do not fall into a local minimum
    p['bs_prior']  = 'all' # 'gs' or 'all': 
                          # randomize prior mean for gs or all priors
    #p['old_bs']    = True # set to True to use BS list from 2009.11825

    p["autotime"]   = 10 # time used to estimate mean gs energy prior
    p["sig_e0"]     = 1 # multiplication factor for meff[autotime] for prior width for deltaE_gs
    p["sig_enn"]    = 1 # multiplication factor for meff[autotime] for prior width for deltaE_nn
    p["positive_z"] = True # force overlaps to be positive or not

    p["ratio"]       = False # fit NN/N1/N2 (True) or NN and N1 N2 (False)
    p["ratio_type"]  = "data" # construct the ratio from the "data" or from ... just use the data
    p["version"]     = 'conspire' # "conspire" or "agnostic"
    p["gs_conspire"] = False # only add deltaE for ground state?
    p["nstates"]     = 3 # number of single nucleon states
    p["r_n_inel"]    = 2 # number of extra NN inelastic states - only relevant for agnostic
    p["r_n_el"]      = 0 # number of extra NN elastic states - only relevant for agnostic
    # pick a range for N and NN (R) for the fit
    p["trange"]      = {"N": [4, 20], "R": [4, 15]}

    p["ampi"] = 0.310810 # ampi is used to construct excited state gaps
    p["amn"]  = 0.70262 # amn is used to estimate elastic excited state gaps
    p["dE_elastic"] = 2 * np.sqrt(p["amn"]**2 + 1 * (2 * np.pi / 48) ** 2) -2*p["amn"]

    # list the irreps and levels in each irrep to fit
    # note, one can fit multiple irreps at once, or individually
    # all irreps in an inner list [("Psq", "irrep", level),()]
    # are simultaneously fit along with the corresponding single
    # nucleon correlators
    if 'deuteron' in p["fpath"]["nn"]:
        p["masterkey"] = [
            [("0", "T1g", 0)], [('0', 'T1g', 1)],
            [('1', 'A2', 0)], [('1', 'A2', 1)], 
            [('1', 'E', 0)], [('1', 'E', 1)], [('4', 'E', 0)], [('4', 'E', 1)],
            [('2', 'A2', 0)], [('4', 'A2', 0)], [('4', 'A2', 1)], 
            [('2', 'B1', 0)], [('2', 'B2', 0)], [('2', 'B2', 3)],
            [('3', 'A2', 0)], [('3', 'A2', 1)], [('3', 'E', 0)]
            ]

        #p["masterkey"] = [[("0", "T1g", 0)]] # modify to select single or other channels

    elif 'dineutron' in p["fpath"]["nn"]:
        p["masterkey"] = []
        for n in range(2):#6):
            p["masterkey"].append([("0", "A1g", n)])
        for n in range(3):#10):
            p["masterkey"].append([("1", "A1", n)])
        for n in range(6):#21):
            p["masterkey"].append([("2", "A1", n)])
        for n in range(3):#9):
            p["masterkey"].append([("3", "A1", n)])
        for n in range(4):#10):
            p["masterkey"].append([("4", "A1", n)])

    return p
