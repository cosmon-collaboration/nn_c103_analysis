#!/usr/bin/env python3
import os
import gvar as gv
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

states = ['0', '1', '2', '3', '4', '5F1', '5F2']
tmin   = range(2,17)
nstate = [1,2,3,4,5]

clrs = {1:'r', 2:'g', 3:'b', 4:'magenta', 5:'orange'}
shft = {1:-0.2, 2:-0.1, 3:0, 4:0.1, 5:0.2}
ylim = {'0': (0.6991,0.709), '1':(0.7121,0.7235), '2':(0.7231,0.7330), '3':(0.7351,0.7450), '4':(0.7451,0.7560), '5F1':(0.7551,0.7670), '5F2':(0.7551,0.7670)}

n_data = gv.load('data/gevp_5_10.pickle')

if not os.path.exists('figures'):
    os.makedirs('figures')

for q in states:
    print(q)
    fig = plt.figure(q,figsize=(7,4))
    ax_e0 = plt.axes([0.12,0.41 ,0.87,0.58])
    ax_Q  = plt.axes([0.12,0.26,0.87,0.15])
    ax_w  = plt.axes([0.12,0.11 ,0.87,0.15])

    q_max = []
    w_max = []
    for t in tmin:
        e = []
        p = []
        w = []
        for n in nstate:
            result = 'result/N_n'+str(n)+'_t_'+str(t)+'-20.pickle'
            if os.path.exists(result):
                data = gv.load(result)
                e.append(data[(q, 'e0')])
                p.append(data[((q,), 'Q')])
                w.append(data[((q,), 'logGBF')])
        e = np.array(e)
        p = np.array(p)
        w = np.array(w)
 
        w = np.exp(w)
        w = w / w.sum()

        q_max.append(max(p))
        w_max.append(max(w))
        for i_n,n in enumerate(nstate[0:len(e)]):
            if t == 2:
                lbl = r'$n=%d$' %n
            else:
                lbl = ''
            ax_e0.errorbar(t+shft[n],e[i_n].mean, yerr=e[i_n].sdev,
                marker='s', color=clrs[n], mfc='None',linestyle='None',label=lbl)
            ax_Q.plot(t+shft[n],p[i_n],marker='s',color=clrs[n],mfc='None',linestyle='None')
            ax_w.plot(t+shft[n],w[i_n],marker='s',color=clrs[n],mfc='None',linestyle='None')

            if n == 2 and t == 7:
                ax_e0.fill_between(np.arange(t,20.5,.5), 
                                   e[i_n].mean-e[i_n].sdev, e[i_n].mean+e[i_n].sdev,
                                   color=clrs[n], alpha=.3)
            if n == 3 and t == 3:
                ax_e0.fill_between(np.arange(t,20.5,.5), 
                                   e[i_n].mean-e[i_n].sdev, e[i_n].mean+e[i_n].sdev,
                                   color=clrs[n], alpha=.3)

    n_corr = n_data[q][2:]
    n_eff  = np.log(n_corr/np.roll(n_corr,-1))
    m  = np.array([k.mean for k in n_eff])
    dm = np.array([k.sdev for k in n_eff])

    ax_e0.errorbar(np.arange(2,2+len(n_eff),1),m,yerr=dm,color='k',marker='s',linestyle='None')
    ax_e0.legend(loc=1, ncol=5,fontsize=16, columnspacing=0,handletextpad=0.1)
    ax_e0.set_ylim(ylim[q][0],ylim[q][0]+0.0099)
    ax_e0.set_ylabel(r'$E_0^{\rm %s}$' %q, fontsize=20)
    ax_w.set_ylim(0,1.2*max(w_max))
    ax_w.set_ylim(0,1.05)
    q_up = 1.01
    if max(q_max) < q_up/2:
        q_up = 2*max(q_max)
    ax_Q.set_ylim(0,q_up)
    ax_Q.set_ylabel(r'$Q$', fontsize=20)
    ax_w.set_ylabel(r'$w$', fontsize=20)
    ax_w.tick_params(bottom=True, top=True, direction='in')
    ax_w.set_xlabel(r'$t_{\rm min}$', fontsize=20)

    ax_e0.set_xlim(1.5,20.5)
    ax_Q.set_xlim(1.5,20.5)
    ax_w.set_xlim(1.5,20.5)

    plt.savefig('figures/nucleon_stability_'+q+'.pdf',transparent=True)


plt.ioff()
plt.show()
