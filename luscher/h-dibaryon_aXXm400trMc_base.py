import numpy as np

def params():
    """
    User input file to control fitting options.

   """
    import numpy as np

    p = dict()
    p["debug"]   = False
    p["verbose"] = False
    p["latex"]   = True

    p["fpath"] = {"nucleon": "./data/nucleon_aXXm400trMc.hdf5", 
                  "nn": "./data/triplet_aXXm400trMc.hdf5"}

    p["save"]   = True
    p["result"] = 'result_aXXm400trMc'

    p["fitter"] = 'scipy_least_squares'

    p["t0"] = 4
    p["td"] = 8
    p['t_norm'] = 3
    p['gevp']   = 'evp' # evp or gevp
    p['get_Zj'] = True
    p['Zjn_values'] = f"result_aXXm400trMc/{p['fpath']['nn'].split('/')[-1].split('_')[0]}"
    p['Zjn_values'] = f"{p['Zjn_values']}_Zjn_tNorm{p['t_norm']}_{p['gevp']}.h5"
    p['show_Zjn']   = False
    p['do_gevp']    = False #set to True if you want to do gevp if it was already done and saved

    p["block"] = 2
    #p['cfgs']  = [0,802[,1]] # use this to cut configs if desired 

    p['svd_study'] = False
    p['svdcut']    = 1e-8

    p["bootstrap"] = False
    p['Nbs_max']   = 5000
    p['bs_seed']   = 'aXXm400trMc_b%d' %p["block"]
    p["nbs"]       = 5000
    p["nbs_sub"]   = 100
    p['bs0_width'] = 5
    p['bs_prior']  = 'all' # 'gs' or 'all': 
                          # randomize prior mean for gs or all priors
    #p['old_bs']    = True # set to True to use BS list from 2009.11825

    p["autotime"]   = ref_time # time used to estimate mean gs energy prior
    p["sig_e0"]     = 1 # multiplication factor for meff[autotime] for prior width for deltaE_gs
    p["sig_enn"]    = 1 # multiplication factor for meff[autotime] for prior width for deltaE_nn
    p["positive_z"] = True

    p["ratio"]       = False
    p["ratio_type"]  = "data"
    p["irreps"]      = "irreps_ben" #["irreps", "irreps_ben"]
    p["version"]     = 'conspire'
    p["gs_conspire"] = False # only add deltaE for ground state?
    p["nstates"]     = 3
    p["r_n_inel"]    = 2
    p["r_n_el"]      = 0
    p["trange"]      = {"N": [3, 15], "R": [3, 12]}

    p["ampi"] = a_mpi
    p["amn"]  = a_mn
    p["dE_elastic"] = 2 * np.sqrt(p["amn"]**2 + 1 * (2 * np.pi / 48) ** 2) -2*p["amn"]

    if 'singlet' in p["fpath"]["nn"]:
        p["masterkey"] = [
            [("0", "T1g", 0)], [('0', 'T1g', 1)],
            [('1', 'A2', 0)], [('1', 'A2', 1)], 
            [('1', 'E', 0)], [('1', 'E', 1)], [('4', 'E', 0)], [('4', 'E', 1)],
            [('2', 'A2', 0)], [('4', 'A2', 0)], [('4', 'A2', 1)], 
            [('2', 'B1', 0)], [('2', 'B2', 0)], [('2', 'B2', 3)],
            [('3', 'A2', 0)], [('3', 'A2', 1)], [('3', 'E', 0)]
            ]

        #p["masterkey"] = [[("0", "T1g", 0)]] # modify to select single or other channels

    elif 'triplet' in p["fpath"]["nn"]:
        p["masterkey"] = []
        for n in range(3):
            p["masterkey"].append([("0", "A1g", n)])
        for n in range(3):
            p["masterkey"].append([("1", "A1", n)])
        for n in range(3):
            p["masterkey"].append([("2", "A1", n)])
        for n in range(3):
            p["masterkey"].append([("3", "A1", n)])
        for n in range(1):
            p["masterkey"].append([("4", "A1", n)])
        #p["masterkey"] = [ [("0", "A1g", 0)]]
        
    return p
