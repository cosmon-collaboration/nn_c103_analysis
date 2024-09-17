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
nn_file  = './result/NN_{nn_iso}_tnorm{tnorm}_t0-td_{gevp}_N_n{N_inel}_t_{N_t}-20_NN_{ver}_e{nn_el}_t_{t0}-15_ratio_{ratio}_block{block}.pickle' #_bsPrior-gs.pickle' #
fig = 'NN_{nn_iso}_full_tnorm{tnorm}_t0-td_{gevp}_N_n{N_inel}_t_{N_t}-20_NN_{ver}_e{nn_el}_t_{t0}-15_ratio_{ratio}.pdf'
best_params = { 'nn_iso': 'triplet', 'tnorm':3,'gevp': "5-10", 'N_inel': 3, 'N_t' : 4 , 'ver' : 'conspire' , 'nn_el' : 0 , 't0' : 5 , 'ratio' : False,  'block' : 2 }
title ='{nn_iso}_tnorm{tnorm}_N_n{N_inel}_t_{N_t}-20_NN_{ver}_e{nn_el}_t_{t0}-15'

shift = { 1:-0.2, 2:-0.1, 4:0., 8:0.1, 16:0.2}
color = { 1:'orange', 2:'r', 4:'g', 8:'b', 16:'magenta'}

params_file = parameters.params()
states = params_file["masterkey"] # calls in the states from params

N_block = [1,2,4,6,18]

print("Starting GEVP Extraction")
gevp_results = {}
ecm_datapath = "./data/ecm_gevp_results.pickle"
for q in states:
    #start_time = time.perf_counter()
    Psq = int(q[0][0])
    level = q[0][2]
    gevp_results[q[0]] = {}