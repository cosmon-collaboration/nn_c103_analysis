import numpy as np

def params():
    """
    User input file to control fitting options.

   """

    p = dict()
    p["debug"]   = False
    p["verbose"] = True
    p["latex"]   = True

    p["fpath"] = {"nucleon": "./data/cls21_c103_r005-8_nucleon_S0.hdf5",'nn':0,'isospin':'singlet'}
    #"nn": "./data/cls21_c103_r005-8_triplet_S0_avg_mom.hdf5"
    if p["fpath"]['isospin'] == 'singlet':
        p["fpath"]['nn'] = "./data/cls21_c103_r005-8_singlet_S0_avg_mom.hdf5"
    elif p["fpath"]['isospin'] == 'triplet':
        p["fpath"]['nn'] = "./data/cls21_c103_r005-8_triplet_S0_avg_mom.hdf5"

    p["save"] = True

    p["fitter"] = 'scipy_least_squares'

    p["t0"] = 5
    p["td"] = 10
    p['t_norm'] = 3

    p['block'] = 1

    p['svd_study'] = False
    p['svdcut']    = 1e-8

    p["bootstrap"] = False
    p['Nbs_max']   = 1000
    p['bs_seed']   = 'nn_c103_b%d' %p["block"]
    p["nbs"]       = 1000
    p["nbs_sub"]   = 100
    p['bs0_width'] = 5
    p['bs_prior']  = 'all'# 'gs' or 'all': 
                          # randomize prior mean for gs or all priors

    p["autotime"]   = 10 # time used to estimate mean gs energy prior
    p["sig_e0"]     = 1  # multiplication factor for meff[autotime] for prior width for deltaE_gs
    p["sig_enn"]    = 1  # multiplication factor for meff[autotime] for prior width for deltaE_nn
    p["positive_z"] = True

    p["ratio"]       = False
    p["ratio_type"]  = "data"
    p["irreps"]      = "irreps_ben" #["irreps", "irreps_ben"]
    p["version"]     = 'conspire'
    p["gs_conspire"] = False # only add deltaE for ground state?
    p["nstates"]     = 3
    p["r_n_inel"]    = 2
    p["r_n_el"]      = 0
    p["trange"]      = {"N": [5, 20], "R": [8, 15]}

    p["ampi"] = 0.310810
    p["amn"]  = 0.70262
    p["dE_elastic"] = 2 * np.sqrt(p["amn"]**2 + 1 * (2 * np.pi / 48) ** 2) -2*p["amn"]

    p['continuum_disp'] = True
    p['vs_mpi'] = False #whether to rescale to mpi for plots
    p['ere_order'] = 2
    p['ere_fit'] = './result/NN_singlet_tnorm3_t0-td_5-10_N_n3_t_3-20_NN_conspire_e0_t_3-15_ratio_False_block2_bsPrior-all.pickle_bs'
    p['irreps_avg'] = False

    if 'singlet' in p["fpath"]["isospin"]:
        p["masterkey"] = []
        for n in range(4): #15
            p["masterkey"].append([("0", "T1g", n)])
        for n in range(4): #10
            p["masterkey"].append([("1", "A2", n)])
        for n in range(4): #18
            p["masterkey"].append([("1", "E", n)])
        for n in range(4): #15
            p["masterkey"].append([("2", "A2", n)])
        for n in range(4): #19
            p["masterkey"].append([("2", "B1", n)])
        for n in range(4): #21
            p["masterkey"].append([("2", "B2", n)])
        for n in range(4): #9
            p["masterkey"].append([("3", "A2", n)])
        for n in range(4): #17
            p["masterkey"].append([("3", "E", n)])
        for n in range(4): #15
            p["masterkey"].append([("4", "E", n)])
        for n in range(4): #7
            p["masterkey"].append([("4", "A2", n)])
        # p["masterkey"] = [
        #     [("0", "T1g", 0)], [('0', 'T1g', 1)],[('0', 'T1g', 2)],
        #     [('1', 'A2', 0)], [('1', 'A2', 1)], [('1', 'A2', 2)], 
        #     [('1', 'E', 0)], [('1', 'E', 1)],  [('1', 'E', 2)],#[('4', 'E', 0)], [('4', 'E', 1)],
        #     [('2', 'A2', 0)], [('2', 'A2', 1)],[('2', 'A2', 2)], #[('4', 'A2', 0)], [('4', 'A2', 1)], 
        #     [('2', 'B1', 0)], [('2', 'B2', 0)], [('2', 'B2', 3)],
        #     [('3', 'A2', 0)], [('3', 'A2', 1)], [('3', 'E', 0)],[('3', 'E', 1)]
        #     ]

        #p["masterkey"] = [[("0", "T1g", 0)]]

    elif 'triplet' in p["fpath"]["isospin"]:
        p["masterkey"] = []
        for n in range(6):#6):
            p["masterkey"].append([("0", "A1g", n)])
        for n in range(10):#10):
            p["masterkey"].append([("1", "A1", n)])
        for n in range(21):#21):
            p["masterkey"].append([("2", "A1", n)])
        for n in range(9):#9):
            p["masterkey"].append([("3", "A1", n)])
        for n in range(10):#10):
            p["masterkey"].append([("4", "A1", n)])
    
    return p
