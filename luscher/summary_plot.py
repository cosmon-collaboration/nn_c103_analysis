import matplotlib.pyplot as plt
import numpy as np
import os

import gvar as gv

def summary_ENN(all_results, mN, all_lbls, colors, lbl0=None, fig='summary', format='pdf'):
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
    irrep_lbls = [
        r'$T_{1g}(0)$', r'$A_2(1)$', r'$E(1)$',
        r'$A_2(2)$', r'$B_1(2)$', r'$B_2(2)$',
        r'$A_2(3)$', r'$E(3)$', r'$A_2(4)$', r'$E(4)$'
    ]
    #import IPython; IPython.embed()
    #plt.ion()
    plt.figure(fig, figsize=(6, 6/1.618))
    ax = plt.axes([.12,.12,.85,.85])
    for k in all_results:
        irrep = (k.split('_')[0], k.split('_')[1])
        DE_i = np.array(all_results[k]['DE'])
        E1_i = np.array(all_results[k]['E1'])
        E2_i = np.array(all_results[k]['E2'])
        nsq  = int(k.split('_')[-1])
        Psq  = nsq * (2*np.pi / 48)**2
        ENN  = DE_i + E1_i + E2_i
        EcmSq= ENN**2 - Psq
        Ecm  = np.sqrt(EcmSq)
        Ecm_mN = Ecm / mN.mean

        for i,e in enumerate(Ecm_mN):
            if irrep in [('0', 'T1g')] and int(k.split('_')[-1]) == 0:
                lbl = all_lbls[i]
                if i == 0:
                    lbl = lbl0+lbl
            else:
                lbl = None
            color = colors[all_lbls[i]]
            ax.errorbar(irreps[irrep]+i-len(Ecm_mN)/2, e.mean, yerr=e.sdev, 
                        marker='s', linestyle='None', mfc='None',
                        color=color, label=lbl)
    ticks = [v for k,v in irreps.items()]
    ax.set_xticks(ticks, labels=irrep_lbls, fontsize=12)
    ax.legend(loc=1,fontsize=12,ncol=len(Ecm_mN), columnspacing=0,handletextpad=0.1)
    ax.set_ylabel(r'$E_{\rm cm} / m_N$', fontsize=16)
    ax.axhline(2, linestyle='--', color='k')
    ax.set_ylim(1.995,2.0651)
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/'+fig+'.'+format, transparent=True)
