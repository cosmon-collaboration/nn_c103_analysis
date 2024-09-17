import os, sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from numpy.linalg import cond
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from scipy import special as scsp
from scipy.interpolate import interp1d

# Peter Lepage libraries
import gvar as gv
import lsqfit

import QC.zeta as zeta
import QC.kinematics as qc_kin
#import IPython; IPython.embed()

import nn_parameters as parameters

params = parameters.params()
isospin = params['fpath']['isospin']
channel = 'deuteron'
# Lattice parameters
L = 48
fitted_states = params["masterkey"]
states = [ [("0", "T1g", 0)], [('0', 'T1g', 1)],[('0', 'T1g', 2)],
          [('1', 'A2', 0)], [('1', 'A2', 1)], [('1', 'A2', 2)], 
          [('1', 'E', 0)], [('1', 'E', 1)],  [('1', 'E', 2)],#[('4', 'E', 0)], [('4', 'E', 1)],
          [('2', 'A2', 0)], [('2', 'A2', 1)],[('2', 'A2', 2)], #[('4', 'A2', 0)], [('4', 'A2', 1)], 
          [('2', 'B1', 0)], [('2', 'B2', 0)], [('2', 'B2', 3)],
          [('3', 'A2', 0)], [('3', 'A2', 1)], [('3', 'E', 0)],[('3', 'E', 1)]
          ]

def load_data():
        fit_results = gv.load(params['ere_fit'])
        # load the nucleon mass
        if channel == 'deuteron':
            mN = np.array(fit_results[((('0', 'T1g', 0), 'N', '0'), 'e0')])
        elif channel == 'dineutron':
            mN = np.array(fit_results[((('0', 'A1g', 0), 'N', '0'), 'e0')])
        
        # load all the NN and N data
        # print("Number of bootstrap samples",len(mN))
        data = {}
        data['mN']  = mN
        print('mean mass',data['mN'][0])
        for state in states:
            #print(state[0])
            state = state[0]
            data[state] = {}
            for k in fit_results:
                if len(k[0]) == 3 and len(k[0][0]) == 3 and (k[0][0] == state and k[0][1] == 'R' and k[1] == 'e0'):
                    Psq, irrep, n = k[0][0] #state form
                    de_nn = np.array(fit_results[k])
                    s1,s2 = k[0][2]
                    st1   = ((k[0][0], 'N', s1), 'e0')
                    st2   = ((k[0][0], 'N', s2), 'e0')
                    en1   = np.array(fit_results[st1])
                    en2   = np.array(fit_results[st2])
                    data[state]['dE_NN'] = de_nn
                    data[state]['Psq']   = int(Psq)
                    data[state]['E_N1']  = en1
                    data[state]['psq_1'] = int(s1)
                    data[state]['E_N2']  = en2
                    data[state]['psq_2'] = int(s2)
        return data

def load_energy_data():
        data = load_data()
        # load the nucleon mass
        mN = data['mN']
        # load all the NN and N data
        print("Number of bootstrap samples",len(mN))
        energy_data = {}
        psq_labels = ['PSQ0','PSQ1','PSQ2','PSQ3']
        for state in data:
            #label = f'PSQ{data[state]['Psq']}'
            energy_data[state] = 
            for k in fit_results:
                if len(k[0]) == 3 and len(k[0][0]) == 3 and (k[0][0] == state and k[0][1] == 'R' and k[1] == 'e0'):
                    Psq, irrep, n = k[0][0] #state form
                    de_nn = np.array(fit_results[k])
                    s1,s2 = k[0][2]
                    st1   = ((k[0][0], 'N', s1), 'e0')
                    st2   = ((k[0][0], 'N', s2), 'e0')
                    en1   = np.array(fit_results[st1])
                    en2   = np.array(fit_results[st2])
                    data[state]['dE_NN'] = de_nn
                    data[state]['Psq']   = int(Psq)
                    data[state]['E_N1']  = en1
                    data[state]['psq_1'] = int(s1)
                    data[state]['E_N2']  = en2
                    data[state]['psq_2'] = int(s2)
        return data

def plot_data(self, ax, data='raw'):
        irrep_clrs = {
            'T1g':'k', 'A2':'b', 'E':'r', 'B1':'g', 'B2':'magenta',
            'A1g':'k', 'A1':'b',
            'A2E':'g', 'A2B1B2':'g',
            }
        level_mrkr = {0:'s', 1:'o', 2:'d', 3:'p', 4:'h', 5:'8', 6:'v'}

        lbl_lst = []
        for state in states:
            # qcotd is really qcotd / mN
            # make qsq dimensionless also
            if data == 'raw':
                qsq_msq  = self.qcotd_raw[state]['qsq'] / self.data['mN']**2
            else:
                qsq_msq  = self.qcotd[state]['qsq'] / self.data['mN']**2
                qcotd    = self.qcotd[state]['qcotd_cm']
            qsq_0    = qsq_msq[0]
            qsq_bs   = qsq_msq[1:]
            qcotd_0  = qcotd[0]
            qcotd_bs = qcotd[1:]
            # shift bs distribution to have boot0 mean
            dqsq     = qsq_bs - qsq_bs.mean()
            qsq_bs   = dqsq + qsq_0
            dqcotd   = qcotd_bs - qcotd_bs.mean()
            qcotd_bs = dqcotd + qcotd_0
            # sort the order for plotting
            i_16   = int(self.Nbs/100*16)
            i_84   = int(self.Nbs/100*84)
            q_sort = qcotd_bs.argsort()
            # plot colors and markers
            irrep  = state[1]
            Psq    = int(state[0])
            mkr    = level_mrkr[Psq]
            spline = interp1d(qsq_bs[q_sort][i_16:i_84], qcotd_bs[q_sort][i_16:i_84], kind='cubic')

            # we only need to make labels with "processed data"
            if data == 'processed':
                lbl = r'${\rm %s}(P_{\rm  tot}^2 = %d)$' %(irrep,Psq)
            else:
                lbl = ''

            qsq_plot = qsq_bs[q_sort][self.Nbs//2]
            # check if data is raw, and also exists in processed data
            alpha = 0.8
            clr   = irrep_clrs[irrep]
            mfc   = clr
            if data == 'raw':
                if state not in self.data_fit:
                    alpha = 0.3
                    mfc   = 'white'
            elif data == 'processed':
                if state not in self.data_fit:
                    alpha = 0.3
                    mfc   = 'white'

            if data == 'raw' or (data=='processed' and irrep not in self.states):
                # don't repeat labels
                if lbl not in lbl_lst and lbl:
                    lbl_lst.append(lbl)
                else:
                    lbl = ''
                ax.plot(qsq_bs[q_sort][i_16:i_84]*self.rescale**2, 
                        spline(qsq_bs[q_sort][i_16:i_84])*self.rescale,
                        color=clr, linewidth=2, alpha=alpha)
                ax.plot(qsq_plot*self.rescale**2, spline(qsq_plot)*self.rescale, 
                        color=clr, label=lbl, marker=mkr, mfc=mfc, alpha=alpha)
def make_qcotd():
    # create qcotd
    self.qcotd     = {}
    self.qcotd_raw = {}
    self.Nbs = self.data['mN'].shape[0] - 1 #remove boot0 to determine Nbs
    print("nbs",self.Nbs)
    for irrep_grp in self.irrep_grps:
        self.qcotd[irrep_grp] = {}
        EN1_w  = 0
        EN2_w  = 0
        dENN_w = 0
        for w,state in self.irrep_grps[irrep_grp]:
            self.qcotd_raw[state] = {}
            if self.params['continuum_disp']:
                EN1 = np.sqrt(self.data['mN']**2 + self.data[state]['psq_1']*(2*np.pi / self.L)**2)
                EN2 = np.sqrt(self.data['mN']**2 + self.data[state]['psq_2']*(2*np.pi / self.L)**2)
            else:
                EN1 = self.data[state]['E_N1']
                EN2 = self.data[state]['E_N2']
            dENN = self.data[state]['dE_NN']

            # populate individual irrep data
            E_NN   = dENN + EN1 + EN2
            E_cmSq = E_NN**2 - self.data[state]['Psq'] * (2 * np.pi / self.L)**2
            E_cm   = np.sqrt(E_cmSq)
            qsq    = E_cmSq / 4 - self.data['mN']**2

            self.qcotd_raw[state]['EcmSq'] = E_cmSq
            self.qcotd_raw[state]['qsq']   = qsq

            Psq   = self.data[state]['Psq']
            irrep = state[1]
            # boxQ  = BMat.BoxQuantization(self.momRay[Psq], Psq, 
            #                             irrep, chanList, 
            #                             [0,], self.Kinv, True)
            # boxQ.setMassesOverRef(0, 1, 1)
            #self.qcotd_raw[state]['qcotd_lab'] = np.zeros(self.Nbs +1)
            self.qcotd_raw[state]['qcotd_cm']  = np.zeros(self.Nbs +1)
            for bs in range(self.Nbs +1):
                ma  = self.data['mN'][bs]/self.data['mN'][bs]
                mb  = self.data['mN'][bs]/self.data['mN'][bs]
                mref = self.data['mN'][bs]
                #boxQ.setRefMassL(self.data['mN'][bs]*self..L)
                #self.qcotd_raw[state]['qcotd_lab'][bs] = boxQ.getBoxMatrixFromElab(E_NN[bs] / self.data['mN'][bs]).real
                self.qcotd_raw[state]['qcotd_cm'][bs]  = qc_kin.qcotd( E_cm[bs]/self.data['mN'][bs], self.L,Psq,1,1,mref)
                #boxQ.getBoxMatrixFromEcm(E_cm[bs] / self.data['mN'][bs]).real

            # for irrep_avg, add weighted contribution
            EN1_w  += w * EN1
            EN2_w  += w * EN2
            dENN_w += w * dENN
        # make irrep_avg data
        E_NN   = dENN_w + EN1_w + EN2_w
        E_cmSq = E_NN**2 - self.data[state]['Psq'] * (2 * np.pi / self.L)**2
        E_cm   = np.sqrt(E_cmSq)
        qsq    = E_cmSq / 4 - self.data['mN']**2

        self.qcotd[irrep_grp]['EcmSq'] = E_cmSq
        self.qcotd[irrep_grp]['qsq']   = qsq

        Psq   = int(irrep_grp[0])
        # we can use either irrep in the irrep_grp to define the BoxMatirx
        irrep = self.irrep_grps[irrep_grp][0][1][1]
        # boxQ  = BMat.BoxQuantization(self.momRay[Psq], Psq, 
        #                             irrep, chanList, 
        #                             [0,], self.Kinv, True)
        #boxQ.setMassesOverRef(0, 1, 1)
        #self.qcotd[irrep_grp]['qcotd_lab'] = np.zeros(self.Nbs+1)
        self.qcotd[irrep_grp]['qcotd_cm']  = np.zeros(self.Nbs+1)
        for bs in range(self.Nbs +1):
            ma  = self.data['mN'][bs]/self.data['mN'][bs]
            mb  = self.data['mN'][bs]/self.data['mN'][bs]
            mref = self.data['mN'][bs]
            #boxQ.setRefMassL(self.data['mN'][bs]*self.args.L)
            #self.qcotd[irrep_grp]['qcotd_lab'][bs] = boxQ.getBoxMatrixFromElab(E_NN[bs] / self.data['mN'][bs]).real
            self.qcotd[irrep_grp]['qcotd_cm'][bs]  = qc_kin.qcotd( E_cm[bs]/self.data['mN'][bs], self.L,Psq,1,1,mref)
            #self.qcotd[irrep_grp]['boxQ'] = lambda x: qc_kin.qcotd( x, self.L,Psq,1,1,self.data['mN'][0])
            #boxQ.getBoxMatrixFromEcm(E_cm[bs] / self.data['mN'][bs]).real
        # set boxQ to boot0 and save
        #boxQ.setRefMassL(self.data['mN'][0]*self.args.L)
        #self.qcotd[irrep_grp]['boxQ'] = qc_kin.qcotd( E_cm[0]/self.data['mN'][0], self.L,Psq,1,1,self.data['mN'][0])



f = load_data()
for state in f:
    print(state)