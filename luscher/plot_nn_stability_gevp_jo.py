#!/usr/bin/env python3
import os, sys, time
import gvar as gv
import numpy as np
import matplotlib.pyplot as plt
# load nn_fit to get fit functions
import nn_fit as fitter
#import argparse
import itertools
import nn_parameters as parameters

from os import path
import pickle


L = 48 

verbose = True

#format = 'pdf'
nn_file  = './result/NN_{nn_iso}_tnorm{tnorm}_t0-td_{gevp}_N_n{N_inel}_t_{N_t}-20_NN_{ver}_e{nn_el}_t_{t0}-15_ratio_{ratio}_block{block}.pickle' #_bsPrior-gs.pickle' # have gevp ranges 
fig = 'NN_{nn_iso}_full_tnorm{tnorm}_t0-td_{gevp}_N_n{N_inel}_t_{N_t}-20_NN_{ver}_e{nn_el}_t_{t0}-15_ratio_{ratio}_block{block}.pdf'
best_params = { 'nn_iso': 'triplet', 'tnorm':3,'gevp': 0, 'N_inel': 3, 'N_t' : 4 , 'ver' : 'conspire' , 'nn_el' : 0 , 't0' : 5 , 'ratio' : False,  'block' : 2 }
title ='{nn_iso}_tnorm{tnorm}_N_n{N_inel}_t_{N_t}-20_NN_{ver}_e{nn_el}_t_{t0}-15_block{block}'

#filename = "./result/NN_{nn}_t0-td_{self.params['t0']}-{self.params['td']}_N_n{self.nstates}_t_{self.params['trange']['N'][0]}-{self.params['trange']['N'][1]}_NN_{self.params['version']}_NN_{self.params['version']}_gs_e{self.r_n_el}_t_{self.params['trange']['R'][0]}-{self.params['trange']['R'][1]}_ratio_{self.params['ratio']}_block{self.block}_bsPrior-{bs_prior}"
# "4-8" "5-10" "5-14" "6-10" "7-13" "8-15"
color = { '3-6' :'yellow', '3-8' :'b', '4-8' :'k', '4-10':'g', 
            '5-10':'r', '5-12':'magenta', '5-14':'aquamarine',
            '6-10':'orange', '6-12':'firebrick','6-14':'b',
            '7-13':'green', '8-15':'violet','9-15':'magenta'}


params_file = parameters.params()
states = params_file["masterkey"] # calls in the states from params
# print(f"States for {best_params['nn_iso']} channel ")
# for i in states:
#     print(i)

#fit_results = gv.load(pa)
      
#for q in states:
#    gevp_results['_'.join([str(k) for k in q])] = {'DE':[],'E1':[],'E2':[]}

# need to get a way to call these when changed 
gevp_times = ["3-6", "3-8", "4-8", "5-10", "5-14", "6-10" ,"6-12", "7-13", "8-15", "9-15"]
#gevp_times = [ '4-8','5-10','5-14', '6-10', '7-13', '8-15']   
#gevp_times = ["3-8" ,  "4-10" , "5-10" ,  "5-12" ,  "6-12"]
gevp_i = {}
# for i in range(len(gevp_times)):
#     n_minus = -1.5
#     if i <= len(gevp_times):
#         gevp_i[gevp_times[i]] = {n_minus + i*0.3}
#     else:
#         gevp_i[gevp_times[i]] = {0 + i*0.3}

# print("gevp_i",gevp_i)

gevp_i = { '3-6':-1.6, '3-8': -1.2,'4-8': -0.8, '5-10':-0.4,  '5-14': 0,
           '6-10': 0.4, '6-12': 0.8, '7-13':1.2, '8-15':1.6, '9-15':2.0}

#gevp_results = {}


print("Starting Ecm Extraction")
gevp_results = {}
ecm_result = {}

ecm_datapath = "./data/ecm_results.pickle"
# datapath = "./data/analysis_gevp_triplet_ecm.pickle"
# if path.exists(datapath):
#     if verbose:
#         print("Reading energy data")

#     gevp_results = gv.load(datapath)
# else:
# if not  path.exists(ecm_datapath):
for q in states:
    #start_time = time.perf_counter()
    Psq = int(q[0][0])
    level = q[0][2]
    gevp_results[q[0]] = {}
    #ecm_result[q[0]] = {}
    for gevp_in in gevp_times:
        gevp_results[q[0]][gevp_in] = {'DE':[],'E1':[],'E2':[]}
        best_params['gevp'] = gevp_in
        file_path = nn_file.format(**best_params)
        fit_result = gv.load(file_path)
        for k in fit_result: #opening the result file to find the energy levels
            #if k[1] == 'e0' and k[0][1] == 'R':
            if len(k[0]) == 3 and len(k[0][0]) == 3 and (k[0][0] == q[0] and k[0][1] == 'R' and k[1] == 'e0'):
                de_nn  = fit_result[k]
                gevp_results[q[0]][gevp_in]['DE'].append(de_nn) 
                s1,s2  = k[0][2]
                st1    = ((k[0][0], 'N', s1), 'e0')
                st2    = ((k[0][0], 'N', s2), 'e0')
                en1    = fit_result[st1]
                en2    = fit_result[st2]
                gevp_results[q[0]][gevp_in]['E1'].append(en1) 
                gevp_results[q[0]][gevp_in]['E2'].append(en2) 
                #print(gevp_results[q[0]][gevp_in])
                #e_nn = de_nn + en1 + en2
                #E_cm = np.sqrt( e_nn**2 - Psq*(2*np.pi/L)**2 )
                #mN = np.array(fit_result[((('0', 'T1g', 0), 'N', '0'), 'e0')])
                #ecm_result[q[0]][gevp_in]= E_cm / mN
                #if verbose:
                #  if gevp_in == '5-10':
                    #   print(f"Energy {q},  gevp={gevp_in}: {ecm_result[q[0]][gevp_in]}")

#gvdata = gv.dataset.avg_data(ecm_result)
# if not os.path.exists(datapath):
#         with open(datapath, "wb") as f:
#             gvdata = gv.dataset.avg_data(**gevp_results)
#             gv.dump(gvdata,f)
            
irreps = {('0', 'A1g'):0,
                  ('1', 'A1') :10,
                  ('2', 'A1') :20,
                  ('3', 'A1') :30,
                  ('4', 'A1') :40
                  }

irrep_lbls = [
            r'$A_{1g}(0)$', r'$A_1(1)$', r'$A_1(2)$', r'$A_1(3)$', r'$A_1(4)$'
        ]
# irrep_lbls = {('0', 'A1g'): '$A_{1g}$ (0)',
#                   ('1', 'A1') :'$A_{1}$ (1)',
#                   ('2', 'A1') :'$A_{1}$ (2)',
#                   ('3', 'A1') :'$A_{1}$ (3)',
#                   ('4', 'A1') :'$A_{1}$ (4)'
#                   }

#gevp_i = { "4-8":-0.9, "5-10": -0.45,"5-14": 0, "6-10":0.45, "7-13": 0.9,"8-15": 0.9}
print(gevp_results)

mN = gevp_results[('0', 'A1g', 0)]['5-10']['E1'][0]

plt.figure(fig, figsize=(6, 6/1.618))
ax = plt.axes([.135,.135,.85,.85])
ecm_result = {} 
for q in states:
    ecm_result[q[0]] = {}
    for gevp_in in gevp_times:
        # dE = gevp_results[q[0]][gevp_in]['DE']
        # E1 = gevp_results[q[0]][gevp_in]['E1']
        nsq = int(q[0][0])
        Psq  = nsq * (2*np.pi / 48)**2
        eNN = gevp_results[q[0]][gevp_in]['DE'][0] + gevp_results[q[0]][gevp_in]['E1'][0] + gevp_results[q[0]][gevp_in]['E2'][0]
        EcmSq = eNN**2 - Psq
        Ecm  = np.sqrt(EcmSq)
        Ecm_mN = Ecm / mN.mean
        #dE_lab = DE_i / mN.mean
        ecm_result[q[0]][gevp_in] = Ecm_mN  # make this

# Save to pickle
# datapath = "./data/ecm_results.pickle"
# with open(datapath, 'wb') as f:
#     pickle.dump(ecm_result, f)

# # Load from pickle
# with open(datapath, 'rb') as f:
#     loaded_ecm_result = pickle.load(f)

# Verify loaded data
#print(loaded_ecm_result)
    
for q in states:
    print(q)
    Psq = q[0][0]
    irrep = q[0][1]
    level = q[0][2]
    irrep_key = (Psq,irrep)
    #if level == 0:
     #   level_energies[irrep_key] = np.empty(0)   
    # now we can plot using the irrep_key to get labels
    # need to stack the levels 
    #method to catch when one energy level is equal to another
    for gevp_in in gevp_times:
        if level == 0 and irrep == 'A1g':
            #if gevp_in == '5-10':
            #    level_energies[irrep_key] = np.append(level_energies[irrep_key], ecm_result[q[0]][gevp_in].mean)
            ax.errorbar( irreps[irrep_key] + gevp_i[gevp_in] , ecm_result[q[0]][gevp_in].mean, yerr = ecm_result[q[0]][gevp_in].sdev,
            marker = 's', linestyle = 'None', mfc = 'None', color = color[gevp_in], label = gevp_in )
            if gevp_i[gevp_in] == 0:
              ax.text(irreps[irrep_key] - 2.2*(-1)**level, ecm_result[q[0]][gevp_in].mean, f"{level+1}", ha='left', va='center', fontsize=7)
        else:
            ax.errorbar( irreps[irrep_key] + gevp_i[gevp_in] , ecm_result[q[0]][gevp_in].mean, yerr = ecm_result[q[0]][gevp_in].sdev,
            marker = 's', linestyle = 'None', mfc = 'None', color = color[gevp_in] )
            if gevp_i[gevp_in] == 0:
              ax.text(irreps[irrep_key] - 2.2*(-1)**level, ecm_result[q[0]][gevp_in].mean, f"{level+1}", ha='left', va='center', fontsize=7)
            # if gevp_in == '5-10':
            #     #if (ecm_result[q[0]][gevp_in].mean - np.any(level_energies[irrep_key]) < .001:
            #     #    ax.text(irreps[irrep_key] + gevp_i["6-12"]+0.1, ecm_result[q[0]][gevp_in].mean, f"{level}", ha='right', va='center', fontsize=12,color='red')
            #     #else:
            #   ax.text(irreps[irrep_key] + gevp_i["5-10"]-1*pow(-1,level), ecm_result[q[0]][gevp_in].mean, f"{level+1}", ha='left', va='center', fontsize=8)
                    
                #level_energies[irrep_key] = np.append(level_energies[irrep_key], ecm_result[q[0]][gevp_in].mean)
 
#ticks = [0,10,20,30,40,50,60,70,80,90]
ticks = [v for k,v in irreps.items()]
ax.set_xticks(ticks, labels=irrep_lbls, fontsize=12)
ax.legend(loc=1, fontsize=8, ncol=2, columnspacing=0,handletextpad=0.1)
ax.set_ylabel(r'$E_{\rm cm} / m_N$', fontsize=16)
ax.axhline(2, linestyle='--', color='k')
ax.set_ylim(1.985,2.195)


# Adjust x-axis limits and spacing
ax.set_xlim(-4, 45) # Adjust the x-axis limits as needed
ax.xaxis.set_tick_params(which='both', pad=15)  # Increase spacing between ticks

plt.savefig('figures/'+fig.format(**best_params), transparent=True)            
            
print("Figure complete ")
                
        
  
  