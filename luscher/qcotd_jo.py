import os, sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py as h5
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


class qsqFit:
    ''' 
        make S-wave only approximation to construct qcotd from energies
        use q_cm**2 to fit parameters of Effective Range Expansion
    '''
    def __init__(self,params,optional_states_list=None):
        # pass CL args to class
        self.params = params
        #self.args = args
        #self.args = args
        self.h5_results = True
        # save fit results
        self.ere_results = {}

        if 'singlet' in self.params["fpath"]["nn"].split('/')[-1]:
            self.channel = 'deuteron'
        elif 'triplet' in self.params["fpath"]["nn"].split('/')[-1]:
            self.channel = 'dineutron'
        else:
            sys.exit(f"your fit_result, {isospin}, is not understood as deuteron or dineutron")

        # for this project, L=48 for all data
        self.L = 48
        # the pion mass is 
        self.mpi = 0.310810

        # what is the set of irreps 
        if self.channel == 'deuteron':
            if not optional_states_list:
                self.states = [
                    # (Psq, irrep, level)
                    ('0', 'T1g', 0),
                    ('0', 'T1g', 1),
                    ('1', 'A2', 0),
                    ('1', 'E', 0),
                    ('1', 'A2', 1),
                    ('1', 'E', 1),
                    ('2', 'A2', 0),
                    ('2', 'B1', 0),
                    ('2', 'B2', 0),
                    ('2', 'B2', 3),
                    ('3', 'A2', 0),
                    ('3', 'E', 0),
                    # ('4', 'A2', 0),
                    # ('4', 'E', 0),
                    # ('4', 'A2', 1),
                    # ('4', 'E', 1),
                ]
                self.irrep_grps = {
                    ('0', 'T1g' , 0):   [(1,   ('0', 'T1g', 0))],
                    ('0', 'T1g' , 1):   [(1,   ('0', 'T1g', 1))],
                    ('1', 'A2E' , 0):   [(1/3, ('1', 'A2', 0)), (2/3, ('1', 'E', 0))],
                    ('1', 'A2E' , 1):   [(1/3, ('1', 'A2', 1)), (2/3, ('1', 'E', 1))],
                    ('3', 'A2'  , 0):   [(1,   ('3', 'A2', 0))],
                    ('3', 'E'   , 0):   [(1,   ('3', 'E', 0))],
                    # ('4', 'A2E' , 0):   [(1/3, ('4', 'A2', 0)), (2/3, ('4', 'E', 0))],
                    # ('4', 'A2E' , 1):   [(1/3, ('4', 'A2', 1)), (2/3, ('4', 'E', 1))],
                    ('2', 'A2B1B2', 0): [(1/3, ('2', 'A2', 0)), (1/3, ('2', 'B1', 0)), (1/3, ('2', 'B2', 0))],
                    ('2', 'B2'  , 3):   [(1,   ('2', 'B2', 3))],
                }
                if not self.params['irreps_avg']:#self.args.irrep_avg:
                    self.irrep_grps = { k:[(1, k)] for k in self.states}
            else:
                self.states = optional_states_list
        elif self.channel == 'dineutron':
            if not optional_states_list:
                self.states = [
                    ("0", "A1g", 0),
                    ("0", "A1g", 1),
                    ("1", "A1", 0),
                    ("1", "A1", 1),
                    ("2", "A1", 0),
                    #("2", "A1", 3),
                    ("3", "A1", 0),
                # ("3", "A1", 1),
                    ("4", "A1", 0),
                    ("4", "A1", 1)
                ]
                self.irrep_grps = {
                    ('0', 'A1g' , 0):   [(1,   ('0', 'A1g', 0))],
                    ('0', 'A1g' , 1):   [(1,   ('0', 'A1g', 1))],
                    ('1', 'A1' , 0):   [(1,   ('1', 'A1', 0))],
                    ('1', 'A1' , 1):   [(1,   ('1', 'A1', 1))],
                    ('2', 'A1' , 0):   [(1,   ('2', 'A1', 0))],
                    #('2', 'A1' , 3):   [(1,   ('2', 'A1', 3))],
                    ('3', 'A1' , 0):   [(1,   ('3', 'A1', 0))],
                    #('3', 'A1' , 1):   [(1,   ('3', 'A1', 1))],
                    ('4', 'A1' , 0):   [(1,   ('4', 'A1', 0))],
                    ('4', 'A1' , 1):   [(1,   ('4', 'A1', 1))],
                }
            #sys.exit('add dineutron states')
        else:
            self.states = optional_states_list

        self.ere_order = self.params['ere_fit_file']
        # load the data
        self.load_data()
        # make qcotd
        self.make_qcotd()
        # prepare data for fit (cut Psq > Psq_max and do irrep averaging)
        self.prepare_data()
 
    def load_data(self):
        fit_results = gv.load(self.params['ere_fit_file'])
        print("fit_results",fit_results)
        # load the nucleon mass
        if self.channel == 'deuteron':
            mN = np.array(fit_results[((('0', 'T1g', 0), 'N', '0'), 'e0')])
        elif self.channel == 'dineutron':
            mN = np.array(fit_results[((('0', 'A1g', 0), 'N', '0'), 'e0')])
        
        # load all the NN and N data
        data = {}
        data['mN']  = mN
        if self.h5_results:
            h5_results = 'data/nn_'+self.channel+'_spectrum.h5'
            print("File:",h5_results)
            with h5.File(h5_results,'w') as f5:
                f5.create_dataset('mN/E0', data=mN)
        for state in self.states:
            print("state",state)
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
                    if self.h5_results:
                        nn_str = irrep+'_Psq'+str(Psq)+'_level'+str(n)
                        print("File nn_Str:",nn_str)
                        with h5.File(h5_results,'a') as f5:
                            f5.create_dataset(nn_str+'/dE_NN',  data=de_nn)
                            f5.create_dataset(nn_str+'/E_N1',   data=en1)
                            f5.create_dataset(nn_str+'/E_N2',   data=en2)
                            f5.create_dataset(nn_str+'/Psq_N1', data=np.array([int(s1)]))
                            f5.create_dataset(nn_str+'/Psq_N2', data=np.array([int(s2)]))
        self.data = data


    def make_qcotd(self):
        # create qcotd
        self.qcotd     = {}
        self.qcotd_raw = {}
        self.Nbs = self.data['mN'].shape[0] - 1 #remove boot0 to determine Nbs
        if self.channel == 'deuteron':
            self.quantum_numbers = {'J': 1,
                                    'S':1,
                                    'l':0}
        elif self.channel == 'dineutron':
            self.quantum_numbers = {'J': 0,
                                    'S':0,
                                    'l':0}

        print("nbs",self.Nbs)
        for irrep_grp in self.irrep_grps:
            self.qcotd[irrep_grp] = {}
            EN1_w  = 0
            EN2_w  = 0
            dENN_w = 0
            for w,state in self.irrep_grps[irrep_grp]:
                self.qcotd_raw[state] = {}
                print(state)
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

    def prepare_data(self):
        ''' cut out data with Psq > Psq max '''
        data_fit = {}
        for state in self.irrep_grps:
            Psq = int(state[0])
            if Psq <= 4:
                data_fit[state] = {}
                data_fit[state]['EcmSq'] = self.qcotd[state]['EcmSq']
                data_fit[state]['qsq']   = self.qcotd[state]['qsq']
                #data_fit[state]['boxQ']  = self.qcotd[state]['boxQ']
        self.data_fit = data_fit

    def ere(self, x, *p):
        ''' qcotd = p[0] + p[1] * x + p[2] * x**2 + ...
        '''
        qcotd = 0.
        for n in range(len(p)):
            qcotd += p[n] * x**n
        return qcotd

    def get_qsq_ere(self, x_dummy, *p):
        results = []
        for k in self.data_fit:
            def residual_sq(x):
                ecm_mn   = 2*np.sqrt(1 + x) # x = qSq / mNSq, 1 = mNSq / mNsq
                Psq = int(k[0])
                qcotd_mn = lambda x: qc_kin.qcotd( x, self.L,Psq,1,1,self.data['mN'][0])
                #self.qcotd[irrep_grp]['boxQ'](ecm_mn)
                #qcotd_mn = self.data_fit[k]['boxQ'].real
                res      = self.ere(x, *p) - qcotd_mn(ecm_mn).real
                return res**2
            
            results.append(least_squares(residual_sq, self.y0[k], method='lm', 
                                        ftol=1.0e-12, gtol=1.0e-12, xtol=1.0e-12).x[0])
        return np.array(results)

    def fit_ere(self,n):
        ''' n: order in q**2 of expansion
            1: q**2
            2: q**4
            qcotd = -1/a + sum_i=1^n c_i (q**2)**n
        '''
        if n == 0:
            p = [0.2]
        if n == 1:
            p = [0.2, 5] 
        if n == 2:
            p = [0.2, 5, -5]
        if n == 3:
            p = [0.2, 5, -5, 5]
        # prepare the data in mN**2 units
        self.y0 = {k:self.data_fit[k]['qsq'][0]/self.data['mN'][0]**2 for k in self.data_fit}
        y_bs    = {k:(self.data_fit[k]['qsq'][1:] - self.data_fit[k]['qsq'][1:].mean())/self.data['mN'][1:]**2 
                   + self.y0[k] for k in self.data_fit}
        self.y_gv = gv.dataset.avg_data(y_bs, bstrap=True)
        y_np   = np.array([self.y0[k] for k in self.y0])
        dy_cov = np.array(gv.evalcov([self.y_gv[k] for k in self.y_gv])) #evaluate covariance matrix
        x_dummy = [n for n in range(len(self.y_gv))]
        # fit the data
        p_opt, p_cov = curve_fit(self.get_qsq_ere, x_dummy, y_np, p0=p, sigma=dy_cov, 
                                 absolute_sigma=True, method='lm')
        p_fit = gv.gvar(p_opt, p_cov)

        r         = self.get_qsq_ere(x_dummy, *p_opt) - y_np
        chisq_min = np.dot(r, np.dot(np.linalg.inv(dy_cov), r))
        dof       = len(y_np) - len(p)

        results = dict()
        results['chisq_dof'] = chisq_min/dof
        results['dof']       = dof
        results['Q']         = scsp.gammaincc(0.5*dof,0.5*chisq_min)
        results['p_opt']     = p_fit

        self.ere_results[n] = results

    def report_results(self, n):
        if self.params['vs_mpi']:
            mpi = 0.310810
            rescale = self.data['mN'][0] / mpi
        else:
            rescale = 1.
        results = self.ere_results[n]
        p = results['p_opt']
        print('------------------------------------------------------------')
        print('ERE fit to O(q_cm**%d)' %(2*(len(p)-1)))
        print('chisq / dof [dof] = %f [%d],  Q = %.2f' 
              %(results['chisq_dof'], results['dof'], results['Q']))
        print('-1/am    = %s' %(p[0] * rescale))
        print('  a m    = %s' %(-1/p[0] / rescale))
        print(' r0 m    = %s' %(2*p[1] / rescale))
        if len(p) >= 3:
            print(' q4 m**3 = %s' %(p[2] / rescale**3))
        if len(p) >= 4:
            print(' q6 m**5 = %s' %(p[3] / rescale**5))

    def plot_qcotd(self):
        ''' make plot of qcotd / M vs qcm**2 / M**2
            M can be either the nucleon mass or the pion mass
            controlled by args.vs_mpi
        '''

        mpi = gv.gvar('0.310810(95)')
        if self.params['vs_mpi']:
            self.rescale = (self.data['mN'][0]/mpi).mean
        else:
            self.rescale = 1.

        plt.figure('qcotd',figsize=(7,4))
        ax = plt.axes([0.12,0.16,0.87,0.83])

        # plot fit results
        fit_clrs = {1:'k', 2:'purple', 3:'blue'}
        qsq      = np.arange(-0.26, 0.53, .001)
        x        = qsq * self.rescale**2
        for n in range(1,self.params['ere_order']+1)[::-1]:
            qcotd = self.ere(qsq, *self.ere_results[n]['p_opt']) * self.rescale
            y  = np.array([k.mean for k in qcotd])
            dy = np.array([k.sdev for k in qcotd])
            ax.fill_between(x, y-dy, y+dy, color=fit_clrs[n], alpha=(5-n)/10)

        # plot the data
        self.plot_data(ax, data='raw')
        # plot the processed data
        self.plot_data(ax, data='processed')

        # set the axes and legend labels
        #ax.legend(loc=2, ncol=5, columnspacing=0, handletextpad=0.1)
        ax.legend(loc=2, columnspacing=0, handletextpad=0.1)
        if self.params['vs_mpi']:
            ax.axis([-.12, 0.26, -.4,1.2])
            ax.set_xlabel(r'$q_{\rm cm}^2 / m_\pi^2$', fontsize=24)
            ax.set_ylabel(r'$q {\rm cot} \delta / m_\pi$', fontsize=24)
        else:
            ax.axis([-.026, 0.0525, -.15,1])
            ax.set_xlabel(r'$q_{\rm cm}^2 / m_N^2$', fontsize=16)
            ax.set_ylabel(r'$q {\rm cot} \delta / m_N$', fontsize=16)
        ax.axhline(color='k')
        ax.axvline(color='k')

        # save the figure
        fig_base = self.params['ere_fit_file'].split('/')[-1]
        # if self.args.fig_type == 'pdf':
        plt.savefig(f'figures/qcotd_{fig_base}.pdf', transparent=True)
        # else:
        #     plt.savefig(f'figures/qcotd_{fig_base}.{self.args.fig_type}')

    def plot_data(self, ax, data='raw'):
        irrep_clrs = {
            'T1g':'k', 'A2':'b', 'E':'r', 'B1':'g', 'B2':'magenta',
            'A1g':'k', 'A1':'b',
            'A2E':'g', 'A2B1B2':'g',
            }
        level_mrkr = {0:'s', 1:'o', 2:'d', 3:'p', 4:'h', 5:'8', 6:'v'}

        if data == 'raw':
            states = self.states
            qcotd  = self.qcotd
        elif data == 'processed':
            states = self.irrep_grps

        lbl_lst = []
        for state in states:
            # qcotd is really qcotd / mN
            # make qsq dimensionless also
            if data == 'raw':
                qsq_msq  = self.qcotd_raw[state]['qsq'] / self.data['mN']**2
                qcotd    = self.qcotd_raw[state]['qcotd_cm']
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

    def plot_data_alt(self,data='raw'):
        plt.figure('qcotd',figsize=(7,4))
        ax = plt.axes([0.12,0.16,0.87,0.83])

        mpi = gv.gvar('0.310810(95)')
        if self.params['vs_mpi']:
            self.rescale = (self.data['mN'][0]/mpi).mean
        else:
            self.rescale = 1.


        irrep_clrs = {
            'T1g':'k', 'A2':'b', 'E':'r', 'B1':'g', 'B2':'magenta',
            'A1g':'k', 'A1':'b',
            'A2E':'g', 'A2B1B2':'g',
            }
        level_mrkr = {0:'s', 1:'o', 2:'d', 3:'p', 4:'h', 5:'8', 6:'v'}

        if data == 'raw':
            states = self.states
            qcotd  = self.qcotd
        elif data == 'processed':
            states = self.irrep_grps
        
        print(states)

        lbl_lst = []
        for state in states:
            # qcotd is really qcotd / mN
            # make qsq dimensionless also
            if data == 'raw':
                qsq_msq  = self.qcotd_raw[state]['qsq'] / self.data['mN']**2
                qcotd    = self.qcotd_raw[state]['qcotd_cm']
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
                lbl = r'${\rm %s}(P_{\rm  tot}^2 = %d)$' %(irrep,Psq)

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
                # if lbl not in lbl_lst and lbl:
                #     lbl_lst.append(lbl)
                # else:
                #     lbl = ''
                # ax.plot(qsq_bs[q_sort][i_16:i_84]*self.rescale**2, 
                #         spline(qsq_bs[q_sort][i_16:i_84])*self.rescale,
                #         color=clr, linewidth=2, alpha=alpha)
                ax.plot(qsq_plot*self.rescale**2, spline(qsq_plot)*self.rescale, 
                        color=clr, label=lbl, marker=mkr, mfc=mfc, alpha=alpha)

            # save the figure
        fig_base = self.params['ere_fit_file'].split('/')[-1]
        # if self.args.fig_type == 'pdf':
        plt.savefig(f'figures/qcotd_{fig_base}.pdf', transparent=True)

                

def main():
    params = parameters.params()
    qsq_fit = qsqFit(params)
    #print(qsq_fit.data)
    ere_order = params['ere_fit_file']
    # plt.ion()
    # fit data
    print("Starting fit")
    print(f"Channel {qsq_fit .channel}")
    print(f"Fit file {params['ere_fit_file']}")
    for n in range(1,params['ere_order']+1):
        qsq_fit.fit_ere(n)
        qsq_fit.report_results(n)

    # plot data
    qsq_fit.plot_qcotd()


    #plt.ioff()
    #plt.show()


if __name__ == "__main__":
    main()


