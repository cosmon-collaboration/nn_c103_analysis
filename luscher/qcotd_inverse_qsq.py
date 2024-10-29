#!/usr/bin/env python3

import os, sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
import h5py as h5
from numpy.linalg import cond
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from scipy import special as scsp
from scipy.interpolate import interp1d

# Peter Lepage libraries
import gvar as gv
import lsqfit

# Box Matrix code of Colin with python binding from Ben
BMat_path='/Users/walkloud/work/research/c51/code/pythib'
sys.path.append(BMat_path)
import BMat

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Fit parameters of an effective range expansion (ERE) to the q_cm**2 values\n'\
                                        +'or to a relativistic model of qcotd = A Ecm + B (Ecm**2 - 4M**2)+...\n'\
                                        +'determined in a fit of the NN and N correlation functions.  This code has one required\n'\
                                        +'argument, which is an input file of an expected form that contains bootstrap resamplings\n'\
                                        +'of the ground state NN and N energy spectrum in various irreps (channels).  An example\n'\
                                        +'of an input file in the Isospin=0, Spin=1 channel is\n\n'\
                                        +'    result/NN_singlet_tnorm3_t0-td_5-10_N_n3_t_4-20_NN_conspire_e0_t_4-15_ratio_False_block8_bsPrior-all.pickle_bs\n\n'\
                                        +'The code works for both the deuteron and di-neutron (Isospin=1, Spin=singlet) irreps.\n'\
                                        +'The optional command line arguments all the user to conrol various features in the analysis.')
    parser.add_argument('fit_result',        help='bootstrap pickle file from analysis')

    parser.add_argument('--ere_order',       default=2, type=int,
                        help=                'max order in ERE expansion, [%(default)s]')
    parser.add_argument('--rel_order',       default=1, type=int,
                        help=                'max order in Rel. qcotd fit [%(default)s]')
    parser.add_argument('--plot_rel',        default=False, action='store_true',
                        help=                'plot Rel. qcotd fits? [%(default)s]')
    parser.add_argument('--irrep_avg',       default=True, action='store_false',
                        help=                'average irreps to suppress physical S-D mixing? [%(default)s]')
    parser.add_argument('--Psq_max',         default=3, type=int,
                        help=                'max value of Psq to use [%(default)s]')
    parser.add_argument('--continuum_disp',  default=True, action='store_false',
                        help=                'use continuum dispersion relation to construct E_NN = dE + E1 + E2? [%(default)s]')
    parser.add_argument('--L',               type=int, default=48, help='spatial box size [%(default)s]')
    parser.add_argument('--Nbs',             type=int, help='set number of bs samples to Nbs')
    parser.add_argument('--plot_hal',        default=False, action='store_true', 
                        help=                'plot HAL potential results? [%(default)s]')
    parser.add_argument('--vs_mpi',          default=True,  action='store_false',
                        help=                'scale qcotd and qsq by mpi? [%(default)s]')
    parser.add_argument('--h5_results',      default=False, action='store_true',
                        help=                'create h5 file with spectrum results? [%(default)s]')
    parser.add_argument('--fig_type',        default='pdf', help='what fig type? [%(default)s]')
    parser.add_argument('--interact',        default=False, action='store_true',
                        help=                'jump to iPython to interact with results? [%(default)s]')

    args = parser.parse_args()
    print(args)

    if args.Nbs:
        sys.exit('This feature of controlling the number of bootstrap samples (Nbs) is not yet supported')

    qsq_fit = qsqFit(args)
    plt.ion()
    # fit data
    for n in range(1,args.ere_order+1):
        qsq_fit.fit_ere(n)
        qsq_fit.report_ere(n)

    for n in range(1,args.rel_order+1):
        qsq_fit.fit_rel_qcotd(n)
        qsq_fit.report_rel_qcotd(n)
    

    # plot data
    qsq_fit.plot_qcotd()

    # interact with fit results?
    if args.interact:
        import IPython; IPython.embed()
    #    
    plt.ioff()
    plt.show()

class qsqFit:
    ''' 
        make S-wave only approximation to construct qcotd from energies
        use q_cm**2 to fit parameters of Effective Range Expansion
    '''

    def __init__(self, args):
        # pass CL args to class
        self.args = args

        # save fit results
        self.ere_results = {}
        self.rel_qcotd_results = {}

        # are we fitting the deuteron or di-neutron channel?
        if 'singlet' in args.fit_result:
            self.channel = 'deuteron'
        elif 'triplet' in args.fit_result:
            self.channel = 'dineutron'
        else:
            sys.exit(f"your fit_result, {args.fit_result}, is not understood as deuteron or dineutron")
        
        # for this project, L=48 for all data
        self.L = 48
        # the pion mass is 
        self.mpi = 0.310810

        # what is the set of irreps 
        if self.channel == 'deuteron':
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
                ('4', 'A2', 0),
                ('4', 'E', 0),
                ('4', 'A2', 1),
                ('4', 'E', 1),
            ]
            self.irrep_grps = {
                ('0', 'T1g' , 0):   [(1,   ('0', 'T1g', 0))],
                ('0', 'T1g' , 1):   [(1,   ('0', 'T1g', 1))],
                ('1', 'A2E' , 0):   [(1/3, ('1', 'A2', 0)), (2/3, ('1', 'E', 0))],
                ('1', 'A2E' , 1):   [(1/3, ('1', 'A2', 1)), (2/3, ('1', 'E', 1))],
                ('3', 'A2'  , 0):   [(1,   ('3', 'A2', 0))],
                ('3', 'E'   , 0):   [(1,   ('3', 'E', 0))],
                ('4', 'A2E' , 0):   [(1/3, ('4', 'A2', 0)), (2/3, ('4', 'E', 0))],
                ('4', 'A2E' , 1):   [(1/3, ('4', 'A2', 1)), (2/3, ('4', 'E', 1))],
                ('2', 'A2B1B2', 0): [(1/3, ('2', 'A2', 0)), (1/3, ('2', 'B1', 0)), (1/3, ('2', 'B2', 0))],
                ('2', 'B2'  , 3):   [(1,   ('2', 'B2', 3))],
            }
            ''' if we are not doing irrep averaging, then make irrep_grps
                with a single element of weight 1 for each state
            '''
            if not self.args.irrep_avg:
                self.irrep_grps = { k:[(1, k)] for k in self.states}
        elif self.channel == 'dineutron':
            self.states = [
                # (Psq, irrep, level)
                ("0", "A1g", 0),
                ("0", "A1g", 1),
                ("1", "A1",  0),
                ("1", "A1",  1),
                ("2", "A1",  0),
                ("2", "A1",  3),
                ("3", "A1",  0),
                ("4", "A1",  0),
                ("4", "A1",  1),
                #("1", "A1",  2),
                #("2", "A1",  1),
                #("2", "A1",  2),
                #("3", "A1",  1),
                #("3", "A1",  2),
            ]
            if self.args.irrep_avg:
                print('\nit does not make sense to irrep avg for di-neutron - ignoring\n')
            self.irrep_grps = { k:[(1, k)] for k in self.states}

        # load the data
        self.load_data()
        # make qcotd
        self.make_qcotd()
        # prepare data for fit (cut Psq > Psq_max and do irrep averaging)
        self.prepare_data()

    def load_data(self):
        fit_results = gv.load(self.args.fit_result)
        # load the nucleon mass
        if self.channel == 'deuteron':
            mN = np.array(fit_results[((('0', 'T1g', 0), 'N', '0'), 'e0')])
        elif self.channel == 'dineutron':
            mN = np.array(fit_results[((('0', 'A1g', 0), 'N', '0'), 'e0')])
        
        # load all the NN and N data
        data = {}
        data['mN']  = mN
        if self.args.h5_results:
            h5_results = 'data/nn_'+self.channel+'_spectrum.h5'
            with h5.File(h5_results,'w') as f5:
                f5.create_dataset('mN/E0', data=mN)
        for state in self.states:
            data[state] = {}
            for k in fit_results:
                if len(k[0]) == 3 and len(k[0][0]) == 3 and (k[0][0] == state and k[0][1] == 'R' and k[1] == 'e0'):
                    Psq, irrep, n = k[0][0]
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
                    if self.args.h5_results:
                        nn_str = irrep+'_Psq'+str(Psq)+'_level'+str(n)
                        with h5.File(h5_results,'a') as f5:
                            f5.create_dataset(nn_str+'/dE_NN',  data=de_nn)
                            f5.create_dataset(nn_str+'/E_N1',   data=en1)
                            f5.create_dataset(nn_str+'/E_N2',   data=en2)
                            f5.create_dataset(nn_str+'/Psq_N1', data=np.array([int(s1)]))
                            f5.create_dataset(nn_str+'/Psq_N2', data=np.array([int(s2)]))
        self.data = data
            


    def make_qcotd(self):
        # BMat functions
        def calcFunc(self, JtimesTwo, Lp, SptimesTwo, chanp, L, 
                     StimesTwo, chan, Ecm_over_mref, pSqFuncList):
            return 0.
        chanList = [BMat.DecayChannelInfo('n','n',1,1,True,True),]

        if self.channel == 'deuteron':
            def isZero(JtimesTwo, Lp, SptimesTwo, chanp, L, StimesTwo, chan):
                return not (JtimesTwo==2 and Lp==0 and L==0 and chanp==0 
                            and chan==0 and SptimesTwo==2 and StimesTwo==2)
            
        elif self.channel == 'dineutron':
            def isZero(JtimesTwo, Lp, SptimesTwo, chanp, L, StimesTwo, chan):
                return not (JtimesTwo==0 and Lp==0 and L==0 and chanp==0 
                            and chan==0 and SptimesTwo==0 and StimesTwo==0)
            
        # define these in self to use in irrep_avg function
        self.momRay = {0 : 'ar', 1 : 'oa', 2 : 'pd', 3 : 'cd', 4 : 'oa' }
        
        self.Kinv = BMat.KMatrix(calcFunc, isZero)

        # create qcotd
        self.qcotd     = {}
        self.qcotd_raw = {}
        if self.args.Nbs:
            self.Nbs = self.args.Nbs
        else:
            self.Nbs = self.data['mN'].shape[0] - 1 #remove boot0 to determine Nbs

        for irrep_grp in self.irrep_grps:
            self.qcotd[irrep_grp] = {}
            EN1_w  = 0
            EN2_w  = 0
            dENN_w = 0
            for w,state in self.irrep_grps[irrep_grp]:
                self.qcotd_raw[state] = {}
                if self.args.continuum_disp:
                    EN1 = np.sqrt(self.data['mN']**2 + self.data[state]['psq_1']*(2*np.pi / self.args.L)**2)
                    EN2 = np.sqrt(self.data['mN']**2 + self.data[state]['psq_2']*(2*np.pi / self.args.L)**2)
                else:
                    EN1 = self.data[state]['E_N1']
                    EN2 = self.data[state]['E_N2']
                dENN = self.data[state]['dE_NN']

                # populate individual irrep data
                E_NN   = dENN + EN1 + EN2
                E_cmSq = E_NN**2 - self.data[state]['Psq'] * (2 * np.pi / self.args.L)**2
                E_cm   = np.sqrt(E_cmSq)
                qsq    = E_cmSq / 4 - self.data['mN']**2

                self.qcotd_raw[state]['EcmSq'] = E_cmSq
                self.qcotd_raw[state]['qsq']   = qsq

                Psq   = self.data[state]['Psq']
                irrep = state[1]
                boxQ  = BMat.BoxQuantization(self.momRay[Psq], Psq, 
                                            irrep, chanList, 
                                            [0,], self.Kinv, True)
                boxQ.setMassesOverRef(0, 1, 1)
                self.qcotd_raw[state]['qcotd_lab'] = np.zeros(self.Nbs +1)
                self.qcotd_raw[state]['qcotd_cm']  = np.zeros(self.Nbs +1)
                for bs in range(self.Nbs +1):
                    boxQ.setRefMassL(self.data['mN'][bs]*self.args.L)
                    self.qcotd_raw[state]['qcotd_lab'][bs] = boxQ.getBoxMatrixFromElab(E_NN[bs] / self.data['mN'][bs]).real
                    self.qcotd_raw[state]['qcotd_cm'][bs]  = boxQ.getBoxMatrixFromEcm(E_cm[bs] / self.data['mN'][bs]).real

                # for irrep_avg, add weighted contribution
                EN1_w  += w * EN1
                EN2_w  += w * EN2
                dENN_w += w * dENN
            # make irrep_avg data
            E_NN   = dENN_w + EN1_w + EN2_w
            E_cmSq = E_NN**2 - self.data[state]['Psq'] * (2 * np.pi / self.args.L)**2
            E_cm   = np.sqrt(E_cmSq)
            qsq    = E_cmSq / 4 - self.data['mN']**2

            self.qcotd[irrep_grp]['EcmSq'] = E_cmSq
            self.qcotd[irrep_grp]['qsq']   = qsq

            Psq   = int(irrep_grp[0])
            # we can use either irrep in the irrep_grp to define the BoxMatirx
            irrep = self.irrep_grps[irrep_grp][0][1][1]
            boxQ  = BMat.BoxQuantization(self.momRay[Psq], Psq, 
                                        irrep, chanList, 
                                        [0,], self.Kinv, True)
            boxQ.setMassesOverRef(0, 1, 1)
            self.qcotd[irrep_grp]['qcotd_lab'] = np.zeros(self.Nbs +1)
            self.qcotd[irrep_grp]['qcotd_cm']  = np.zeros(self.Nbs +1)
            for bs in range(self.Nbs +1):
                boxQ.setRefMassL(self.data['mN'][bs]*self.args.L)
                self.qcotd[irrep_grp]['qcotd_lab'][bs] = boxQ.getBoxMatrixFromElab(E_NN[bs] / self.data['mN'][bs]).real
                self.qcotd[irrep_grp]['qcotd_cm'][bs]  = boxQ.getBoxMatrixFromEcm(E_cm[bs] / self.data['mN'][bs]).real
            # set boxQ to boot0 and save
            boxQ.setRefMassL(self.data['mN'][0]*self.args.L)
            self.qcotd[irrep_grp]['boxQ'] = boxQ

    def prepare_data(self):
        ''' cut out data with Psq > Psq max '''
        data_fit = {}
        for state in self.irrep_grps:
            Psq = int(state[0])
            if Psq <= self.args.Psq_max:
                data_fit[state] = {}
                data_fit[state]['EcmSq'] = self.qcotd[state]['EcmSq']
                data_fit[state]['qsq']   = self.qcotd[state]['qsq']
                data_fit[state]['boxQ']  = self.qcotd[state]['boxQ']
        self.data_fit = data_fit

    def fit_ere(self,n):
        ''' This function finds the values of the parameters of the ERE 
            that minimize the correlated values of qcm**2 with respect to
            the predicted values of qcm**2 using the Luscher quantization condition (QC)

            It uses get_qsq_ere(), defined below, to find qcm**2 through the QC.
        
            It takes in the order, n, of the ERE to perform the fit
            n: order in q**2 of expansion
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
        y_bs    = {k:(self.data_fit[k]['qsq'][1:self.Nbs] - self.data_fit[k]['qsq'][1:self.Nbs].mean())/self.data['mN'][1:self.Nbs]**2 
                   + self.y0[k] for k in self.data_fit}
        self.y_gv = gv.dataset.avg_data(y_bs, bstrap=True)
        y_np    = np.array([self.y0[k] for k in self.y0])
        dy_cov  = np.array(gv.evalcov([self.y_gv[k] for k in self.y_gv]))
        x_dummy = [n for n in range(len(self.y_gv))]

        # fit the data
        p_opt, p_cov = curve_fit(self.get_qsq_ere, x_dummy, y_np, p0=p, sigma=dy_cov, 
                                 absolute_sigma=True, method='lm')
        p_fit = gv.gvar(p_opt, p_cov)

        # given fit, build chisq minimum
        r         = self.get_qsq_ere(x_dummy, *p_opt) - y_np
        chisq_min = np.dot(r, np.dot(np.linalg.inv(dy_cov), r))
        dof       = len(y_np) - len(p)

        results = dict()
        results['chisq_dof'] = chisq_min/dof
        results['dof']       = dof
        results['Q']         = scsp.gammaincc(0.5*dof,0.5*chisq_min)
        results['p_opt']     = p_fit

        self.ere_results[n] = results

    def ere(self, x, *p):
        ''' x = qcm**2 / MN**2
            qcotd = p[0] + p[1] * x + p[2] * x**2 + ...
        '''
        qcotd = 0.
        for n in range(len(p)):
            qcotd += p[n] * x**n
        return qcotd

    def get_qsq_ere(self, x_dummy, *p):
        ''' This function is designed to return the values of x=qcm**2/M**2,
            for a fixed set of parameters, p, that minimize the residual

            r = ere(x, *p) - qcotd_mn

            where qcotd_mn is the value of qcodt given the QC in MN units
        '''
        results = []
        for k in self.data_fit:
            def residual_sq(x):
                ''' x = qSq / mNSq 
                    ecm will be in units of MN
                '''
                ecm_mn   = 2*np.sqrt(1 + x)
                qcotd_mn = self.data_fit[k]['boxQ'].getBoxMatrixFromEcm(ecm_mn).real
                res      = self.ere(x, *p) - qcotd_mn
                return res**2
            
            results.append(least_squares(residual_sq, self.y0[k], method='lm', 
                                         ftol=1.0e-12, gtol=1.0e-12, xtol=1.0e-12).x[0])
        return np.array(results)


    def report_ere(self, n):
        if self.args.vs_mpi:
            rescale = self.data['mN'][0] / self.mpi
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

    def fit_rel_qcotd(self,n):
        ''' fit qcotdelta to relativistic forms
            A * Ecm + B * Ecm * (Ecm**2 - 4) + C * Ecm * (Ecm**2 - 4)**2 ...
            where
            Ecm**2 = 4( MN**2 + qcm**2) / MN**2

            the code mimics the ERE fit above, and works to minimize values of
            x = qcm**2 / MN**2

            to optimize the parameters
        '''
        if n == 0:
            p = [0.05]
        elif n == 1:
            p = [0.05, 1]
        elif n == 2:
            p = [0.05, 1, 1]
        else:
            sys.exit('add higher order (than n=1) fit params')
        # prepare the data in mN**2 units
        self.y0 = {k:self.data_fit[k]['qsq'][0]/self.data['mN'][0]**2 for k in self.data_fit}
        y_bs    = {k:(self.data_fit[k]['qsq'][1:self.Nbs] - self.data_fit[k]['qsq'][1:self.Nbs].mean())/self.data['mN'][1:self.Nbs]**2 
            + self.y0[k] for k in self.data_fit}
        self.y_gv = gv.dataset.avg_data(y_bs, bstrap=True)
        y_np   = np.array([self.y0[k] for k in self.y0])
        dy_cov = np.array(gv.evalcov([self.y_gv[k] for k in self.y_gv]))
        x_dummy = [n for n in range(len(self.y_gv))]
        # fit the data
        p_opt, p_cov = curve_fit(self.get_rel_qcotd, x_dummy, y_np, p0=p, sigma=dy_cov, 
                                 absolute_sigma=True, method='lm')
        p_fit = gv.gvar(p_opt, p_cov)

        r         = self.get_rel_qcotd(x_dummy, *p_opt) - y_np
        chisq_min = np.dot(r, np.dot(np.linalg.inv(dy_cov), r))
        dof       = len(y_np) - len(p)

        results = dict()
        results['chisq_dof'] = chisq_min/dof
        results['dof']       = dof
        results['Q']         = scsp.gammaincc(0.5*dof,0.5*chisq_min)
        results['p_opt']     = p_fit

        self.rel_qcotd_results[n] = results


    def rel_qcotd(self, x, *p):
        ''' x     = qSq / mNSq
            Ecm   = 2 * np.sqrt(1 + x)
            qcotd = p[0] * Ecm + p[1] * Ecm * (Ecm**2 - 4) + p[2] * Ecm * (Ecm**2 - 4)**2 + ...
        '''
        Ecm   = 2 * np.sqrt(1 + x)
        qcotd = p[0] * Ecm
        if len(p) >= 2:
            qcotd += p[1] * Ecm * 4 * x
        if len(p) >= 3:
            qcotd += p[2] * Ecm * (4 * x)**2
        if len(p) >= 4:
            sys.exit('we only support up to 2nd order')

        return qcotd

    def get_rel_qcotd(self, x_dummy, *p):
        results = []
        for k in self.data_fit:
            def residual_sq(x):
                ecm_mn   = 2*np.sqrt(1 + x) # x = qSq / mNSq
                qcotd_mn = self.data_fit[k]['boxQ'].getBoxMatrixFromEcm(ecm_mn).real
                res      = self.rel_qcotd(x, *p) - qcotd_mn
                return res**2
            
            results.append(least_squares(residual_sq, self.y0[k], method='lm', 
                                        ftol=1.0e-12, gtol=1.0e-12, xtol=1.0e-12).x[0])
        return np.array(results)

    def report_rel_qcotd(self, n):
        if self.args.vs_mpi:
            rescale = self.data['mN'][0] / self.mpi
        else:
            rescale = 1.
        results = self.rel_qcotd_results[n]
        p = results['p_opt']
        print('------------------------------------------------------------')
        print('Rel. qcotd fit to O(q_cm**%d)' %(2*(len(p)-1)+1))
        print('chisq / dof [dof] = %f [%d],  Q = %.2f' 
              %(results['chisq_dof'], results['dof'], results['Q']))
        print(' A      = %s' %(p[0] * rescale))
        print(' B m**2 = %s' %(p[1] / rescale**2))
        if len(p) >= 3:
            print(' C m**3 = %s' %(p[2] / rescale**4))


    def plot_qcotd(self):
        ''' make plot of qcotd / M vs qcm**2 / M**2
            M can be either the nucleon mass or the pion mass
            controlled by args.vs_mpi
        '''

        if self.args.vs_mpi:
            self.rescale = (self.data['mN'][0]/self.mpi)
        else:
            self.rescale = 1.
        plt.figure('qcotd',figsize=(7,4))
        ax = plt.axes([0.12,0.16,0.87,0.83])

        # plot fit results
        fit_clrs = {1:'r', 2:'g', 3:'b'}
        qsq      = np.arange(-0.26, 0.53, .001)
        x        = qsq * self.rescale**2
        for n in range(1,self.args.ere_order+1)[::-1]:
            qcotd = self.rescale * self.ere(qsq, *self.ere_results[n]['p_opt'])
            y  = np.array([k.mean for k in qcotd])
            dy = np.array([k.sdev for k in qcotd])
            ax.fill_between(x, y-dy, y+dy, color=fit_clrs[n], alpha=(5-n)/10)

        if self.args.plot_rel:
            # plot Rel. qcotd
            fit_clrs = {1:'yellow', 2:'magenta'}
            for n in range(1,self.args.rel_order+1)[::-1]:
                qcotd = self.rescale * self.rel_qcotd(qsq, *self.rel_qcotd_results[n]['p_opt'])
                y  = np.array([k.mean for k in qcotd])
                dy = np.array([k.sdev for k in qcotd])
                ax.plot(x, y-dy, color=fit_clrs[n], linestyle='--')
                ax.plot(x, y+dy, color=fit_clrs[n], linestyle='--')

        # plot the data
        #self.plot_data(ax, data='raw')
        # plot the processed data
        self.plot_data(ax, data='processed')

        if self.args.plot_hal:
            '''
            # plot HAL QCD results
            '''
            if self.channel == 'deuteron':
                hal_in = open('data/pcotd_t13.txt').readlines()
            elif self.channel == 'dineutron':
                hal_in = open('data/pcotd_nn_t13.txt').readlines()
            qsq_hal = []
            hal_results = []
            for i in range(24):
                tmp_hal = []
                for qi in range(21):
                    if i == 0:
                        qsq_hal.append(float(hal_in[i*22+qi].split(',')[0]))
                    tmp_hal.append(gv.gvar(hal_in[i*22+qi].split(',')[1]))
                hal_results.append(tmp_hal)
            qsq_hal = np.array(qsq_hal)
            hal_results = np.array(hal_results)
            dhal = []
            for i in range(hal_results.shape[1]):
                dhal.append(max([k.sdev for k in hal_results[:,i]]))
            dhal = np.array(dhal)
            hal_mean = np.array([k.mean for k in hal_results.mean(axis=0)])
            hal_var  = np.array([k.mean for k in (hal_results**2 - hal_mean**2).mean(axis=0)])
            dhal_tot = np.sqrt(dhal**2 + hal_var)

            #import IPython; IPython.embed()
            ax.fill_between(qsq_hal, hal_mean-dhal_tot, hal_mean+dhal_tot, color='k', 
                            alpha=0.3, label='HAL QCD Potential')
            #ax.fill_between(qsq_hal, hal_mean-dhal, hal_mean+dhal, color='k', alpha=0.3)

        # set the axes and legend labels
        #ax.legend(loc=2, ncol=5, columnspacing=0, handletextpad=0.1)
        ax.legend(loc=2, columnspacing=0, handletextpad=0.1)
        if self.args.vs_mpi:
            if self.channel == 'deuteron':
                ax.axis([-.12, 0.26, -.4,1.2])
            elif self.channel == 'dineutron':
                ax.axis([-.12, 0.26, -.4,2.0])
            ax.set_xlabel(r'$q_{\rm cm}^2 / m_\pi^2$', fontsize=24)
            ax.set_ylabel(r'$q {\rm cot} \delta / m_\pi$', fontsize=24)
        else:
            if self.channel == 'deuteron':
                ax.axis([-.026, 0.0525, -.15,0.6])
            elif self.channel == 'dineutron':
                ax.axis([-.026, 0.0525, -.15,1.0])
            ax.set_xlabel(r'$q_{\rm cm}^2 / m_N^2$', fontsize=16)
            ax.set_ylabel(r'$q {\rm cot} \delta / m_N$', fontsize=16)
        ax.axhline(color='k')
        ax.axvline(color='k')

        # save the figure
        fig_base = self.args.fit_result.split('/')[-1]
        if self.args.fig_type == 'pdf':
            plt.savefig(f'figures/qcotd_{fig_base}.pdf', transparent=True)
        else:
            plt.savefig(f'figures/qcotd_{fig_base}.{self.args.fig_type}')

    def plot_data(self, ax, data='raw'):
        if self.channel == 'deuteron':
            irrep_clrs = {
                'T1g':'k', 'A2':'b', 'E':'r', 'B1':'g', 'B2':'magenta',
                'A1g':'k', 'A1':'b',
                'A2E':'g', 'A2B1B2':'g',
                }
        elif self.channel == 'dineutron':
            irrep_clrs = {
                'A1g-0':'k',
                'A1-1' :'r',
                'A1-2' :'g',
                'A1-3' :'b',
                'A1-4' :'magenta'
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
            qsq_bs   = qsq_msq[1:self.Nbs]
            qcotd_0  = qcotd[0]
            qcotd_bs = qcotd[1:self.Nbs]
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
            if self.channel == 'deuteron':
                clr   = irrep_clrs[irrep]
            elif self.channel == 'dineutron':
                clr   = irrep_clrs[irrep+'-'+str(Psq)]
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

    
if __name__ == "__main__":
    main()
