import matplotlib.pyplot as plt
import numpy as np
import os

import gvar as gv

def summary_ENN(all_results, mN, all_lbls, colors, lbl0=None, spin='singlet', fig='summary', format='pdf'):
    if spin == 'singlet':
        irreps = {('0', 'T1g'):0,
                ('1', 'A2') :10, 
                ('1', 'E')  :20,
                ('2', 'A2') :30, 
                ('2', 'B1') :40,  
                ('2', 'B2') :50,
                ('3', 'A2') :60, 
                ('3', 'E')  :70,
                ('4', 'A2') :80, 
                ('4', 'E')  :90
                }
    elif spin == 'triplet':
        irreps = {('0', 'A1g'):0,
                  ('1', 'A1') :10,
                  ('2', 'A1') :20,
                  ('3', 'A1') :30,
                  ('4', 'A1') :40
                  }
    if spin == 'singlet':
        irrep_lbls = [
            r'$T_{1g}(0)$', r'$A_2(1)$', r'$E(1)$',
            r'$A_2(2)$', r'$B_1(2)$', r'$B_2(2)$',
            r'$A_2(3)$', r'$E(3)$', r'$A_2(4)$', r'$E(4)$'
        ]
    elif spin == 'triplet':
        irrep_lbls = [
            r'$A_{1g}(0)$', r'$A_1(1)$', r'$A_1(2)$', r'$A_1(3)$', r'$A_1(4)$'
        ]
    #import IPython; IPython.embed()
    #plt.ion()
    plt.figure(fig, figsize=(6, 6/1.618))
    ax = plt.axes([.135,.135,.85,.85])
    plt.figure(fig+'_dElab', figsize=(6, 6/1.618))
    ax_dE = plt.axes([.135,.135,.85,.85])
    for k in all_results:
        irrep = (k.split('_')[0], k.split('_')[1])
        DE_i = np.array(all_results[k]['DE'])
        E1_i = np.array(all_results[k]['E1'])
        E2_i = np.array(all_results[k]['E2'])
        nsq  = int(k.split('_')[0])
        Psq  = nsq * (2*np.pi / 48)**2
        ENN  = DE_i + E1_i + E2_i
        EcmSq= ENN**2 - Psq
        Ecm  = np.sqrt(EcmSq)
        Ecm_mN = Ecm / mN.mean
        dE_lab = DE_i / mN.mean

        for i,e in enumerate(Ecm_mN):
            if irrep in [('0', 'T1g'), ('0', 'A1g')] and int(k.split('_')[-1]) == 0:
                lbl = str(all_lbls[i])
                if i == 0:
                    lbl = lbl0 + lbl
            else:
                lbl = None
            color = colors[all_lbls[i]]
            ax.errorbar(irreps[irrep]+i-len(Ecm_mN)/2, e.mean, yerr=e.sdev, 
                        marker='s', linestyle='None', mfc='None',
                        color=color, label=lbl)
            
            ax_dE.errorbar(irreps[irrep]+i-len(Ecm_mN)/2, dE_lab[i].mean, yerr=dE_lab[i].sdev, 
                        marker='s', linestyle='None', mfc='None',
                        color=color, label=lbl)
    ticks = [v for k,v in irreps.items()]
    ax.set_xticks(ticks, labels=irrep_lbls, fontsize=12)
    ax.legend(loc=1,fontsize=12,ncol=len(Ecm_mN), columnspacing=0,handletextpad=0.1)
    ax.set_ylabel(r'$E_{\rm cm} / m_N$', fontsize=16)
    ax.axhline(2, linestyle='--', color='k')
    ax.set_ylim(1.995,2.0551)
    plt.figure(fig)
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/'+fig+'.'+format, transparent=True)

    ax_dE.set_xticks(ticks, labels=irrep_lbls, fontsize=12)
    ax_dE.legend(loc=1,fontsize=12,ncol=len(Ecm_mN), columnspacing=0,handletextpad=0.1)
    ax_dE.set_ylabel(r'$\Delta E_{\rm lab} / m_N$', fontsize=16)
    ax_dE.set_ylim(-0.009,0)
    plt.figure(fig+'_dElab')
    plt.savefig('figures/'+fig+'_dElab'+'.'+format, transparent=True)
    
