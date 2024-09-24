import os, sys
import numpy as np
import argparse
import matplotlib.pyplot as plt

from numpy.linalg import cond
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from scipy import special as scsp
from scipy.interpolate import interp1d
from pylatex import Document, Section, Tabular, NoEscape

# Peter Lepage libraries
import gvar as gv
import lsqfit

import QC.zeta as zeta
import QC.kinematics as qc_kin
#import IPython; IPython.embed()

import nn_parameters as parameters



class qsqTables:
    ''' 
        make S-wave only approximation to construct qcotd from energies
        use q_cm**2 to fit parameters of Effective Range Expansion
    '''
    def __init__(self,params):
        # pass CL args to class
        self.params = params
        #self.args = args

        # save fit results
        self.table_results = {}
        isospin = self.params['fpath']['isospin']
        # are we fitting the deuteron or di-neutron channel?
        if 'singlet' in isospin:
            self.channel = 'deuteron'
        elif 'triplet' in isospin:
            self.channel = 'dineutron'
        else:
            sys.exit(f"your fit_result, {isospin}, is not understood as deuteron or dineutron")
        # for this project, L=48 for all data
        self.L = 48
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
            if not self.params['irreps_avg']:#self.args.irrep_avg:
                self.irrep_grps = { k:[(1, k)] for k in self.states}
        elif self.channel == 'dineutron':
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
        # load the data
        self.load_table_data()
        self.print_table_data()

    def load_table_data(self):
        fit_results = gv.load(self.params['ere_fit'])
        # load the nucleon mass
        if self.channel == 'deuteron':
            mN = np.array(fit_results[((('0', 'T1g', 0), 'N', '0'), 'e0')])
        elif self.channel == 'dineutron':
            mN = np.array(fit_results[((('0', 'A1g', 0), 'N', '0'), 'e0')])
        
        # load all the NN and N data
        data = {}
        data['mN']  = mN
        print('mean nucleon mass',data['mN'][0])
        for state in self.states:
            data[state] = {}
            for k in fit_results:
                if len(k[0]) == 3 and len(k[0][0]) == 3 and (k[0][0] == state and k[0][1] == 'R' and k[1] == 'e0'):
                    Psq, irrep, n = k[0][0] #state form
                    data[state]['$d^2$'] = Psq
                    data[state]['Irrep'] = irrep
                    data[state]['State'] = n
                    Psq = int(Psq)
                    de_nn = np.array(fit_results[k])
                    data[state]['dE_NN'] = de_nn
                    s1,s2 = k[0][2]
                    data[state]['$d_1$'] = int(s1)
                    data[state]['$d_2$'] = int(s2)
                    st1   = ((k[0][0], 'N', s1), 'e0')
                    st2   = ((k[0][0], 'N', s2), 'e0')
                    en1   = np.array(fit_results[st1])
                    en2   = np.array(fit_results[st2])
                    data[state]['E_N1']  = en1
                    data[state]['E_N2']  = en2
                    EN1 = np.sqrt(data['mN']**2 + int(s1)*(2*np.pi / self.L)**2)
                    EN2 = np.sqrt(data['mN']**2 + int(s2)*(2*np.pi / self.L)**2)
                    E_NN   = de_nn + EN1 + EN2
                    data[state]['E_NN']  = E_NN
                    E_cmSq = E_NN**2 - Psq * (2 * np.pi / self.L)**2
                    E_cm   = np.sqrt(E_cmSq)
                    data[state]['E_cm']  = E_cm
                    qsq    = E_cmSq / 4 - data['mN']**2
                    data[state]['qsq']  = qsq
                    qcotd_raw  = qc_kin.qcotd( E_cm/data['mN'], self.L,Psq,1,1,data['mN'] )
                    data[state]['qcotd']  = qcotd_raw 
                    
        self.table_data = data
    def create_table(self):
        # Create a LaTeX document
        doc = Document()
        # Add a section
        with doc.create(Section('Table')):
            # Create a tabular environment
            with doc.create(Tabular('|' + 'c|' * len(self.table_data) + '|')) as table:
                table.add_hline()  # Horizontal line

                # Add header row
                headers = list(self.table_data.keys())
                table.add_row(headers)
                table.add_hline()

                # Add data rows
                for i in range(len(list(self.table_data.values())[0])):
                    row = [str(self.table_data[key][i]) for key in headers]
                    table.add_row(row)
                    table.add_hline()
                table.add_hline()
        # Generate LaTeX document
        fig_base = self.params['ere_fit'].split('/')[-1]
        doc.generate_pdf(f'./result/table_data_{fig_base}', clean_tex=False)


def main():
    params = parameters.params()
    qsq_fit = qsqTables(params)
    #print(qsq_fit.data)]
    # plt.ion()
    # fit data
    print("Saving tables")
    # plot data
    qsq_fit.create_table()


    #plt.ioff()
    #plt.show()


if __name__ == "__main__":
    main()












#prepare dict with data needed for tables