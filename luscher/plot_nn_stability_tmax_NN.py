#!/usr/bin/env python3
import os, sys, time
import gvar as gv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools
#
# load nn_fit to get fit functions
import nn_fit as fitter
import summary_plot

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

def main():
    parser = argparse.ArgumentParser(
        description='Plot results from NN fits with various correlator models')
    # optimal fit 
    parser.add_argument('optimal',  type=str, 
                        help=       'add optimal fit result')

    # NN info
    parser.add_argument('--n_N',    nargs='+', type=int, default=[3],
                        help=       'number of exponentials in single nucleon to sweep over %(default)s')
    parser.add_argument('--nn_el',  nargs='+', default=[0],
                        help=       'number of elastic e.s. to try %(default)s')
    parser.add_argument('--ratio',  default=False, action='store_true',
                        help=       'fit from RATIO correlator? [%(default)s]')
    parser.add_argument('--tf_NN',  nargs='+', type=int, default=[10, 11, 12, 13, 14, 15],
                        help=       'list of t0_min times for single nucleon %(default)s')
    parser.add_argument('--tmin',   nargs='+', type=int, default=[3,4,5,6,7],
                        help=       'values of t_min in NN fit [%(default)s]')
    parser.add_argument('--evp',    default=True, action='store_false',
                        help=       'load evp (vs gevp) data? [%(default)s]')
    parser.add_argument('--gs_cons',default=False, action='store_true',
                        help=       'use gs only conspiracy model? [%(default)s]')
    parser.add_argument('--test',   default=False, action='store_true',
                        help=       'if test==True, only do T1g [%(default)s]')
    
    parser.add_argument('--debug',  default=False, action='store_true',
                        help=       'add extra debug print statements? [%(default)s]')
   
    args = parser.parse_args()
    print(args)

    color = { 10:'orange', 11:'r', 12:'g', 13:'b', 14:'magenta', 15:'gray' }

    if 'block' in args.optimal:
        block = '_block' + args.optimal.split('block')[1].split('_')[0].split('.')[0]
    else:
        block = ''

    N_t = args.optimal.split('_NN')[0].split('_')[-1]

    nn_file  = 'NN_{nn_iso}_{t_norm}_t0-td_{gevp}_N_n{N_inel}_t_{N_t}'
    nn_file += '_NN_{nn_model}_e{nn_el}_t_{t0}-{tf}_ratio_'+str(args.ratio)+block
    if 'bsPrior' in args.optimal:
        bsPrior = args.optimal.split('bsPrior-')[1].split('.')[0]
        nn_file += '_bsPrior-'+bsPrior
    else:
        bsPrior = ''
    nn_file += '.pickle'

    nn_iso    = args.optimal.split('/')[-1].split('_')[1]
    tnorm     = args.optimal.split('/')[-1].split('_')[2]
    gevp_plot = args.optimal.split('t0-td_')[1].split('_')[0]

    nn_dict = { 'N_t':N_t, 't_norm':tnorm, 'nn_iso':nn_iso, }
    nn_dict.update({'gevp':gevp_plot})

    nn_model = 'N_n{N_inel}_NN_{nn_model}_e{nn_el}'

    models = {}
    for n in args.n_N:
        models['N_n%d_NN_conspire_e0' %n] = {'N_inel':n, 'nn_model':'conspire', 'nn_el':0}
    
    if not os.path.exists("figures"):
        os.mkdir("figures")

    states = summary_plot.get_states(nn_iso, args.test)

    print('\nloading optimal fit:',args.optimal)
    post_optimal  = gv.load(args.optimal)
    optimal_p     = {'positive_z':True, 'debug':False}
    N_inel        = int(args.optimal.split('N_n')[1].split('_')[0])
    nn_el         = int(args.optimal.split('_e')[1].split('_')[0])
    optimal_model = {}
    if 'conspire' in args.optimal:
        optimal_p['version']      = 'conspire'
        optimal_model['nn_model'] = 'conspire'
    else:
        sys.exit('you supplied an "agnostic" model, but we require "conspire"')
    
    optimal_p['r_n_el']      = nn_el
    optimal_p['nstates']     = N_inel
    optimal_p['gs_conspire'] = args.gs_cons
    optimal_p['bs_prior']    = bsPrior
    optimal_model['N_inel']  = N_inel
    optimal_model['nn_el']   = nn_el

    optimal_model = nn_model.format(**optimal_model)

    opt_tmin  = int(args.optimal.split('_NN_')[1].split('-')[0].split('_')[-1])

    t0_plot   = int(args.optimal.split('_t_')[-1].split('_ratio')[0].split('-')[1])
    opt_clr   = color[t0_plot]

    # get data keys
    fit_keys = {}
    for q in states:
        for k in post_optimal:
            if k[1] == 'e0' and k[0][1] == 'R' and k[0][0] == q:
                fit_keys[q] = k

    if args.evp:
        d_file = f"data/gevp_{nn_iso}_{tnorm}_evp_{gevp_plot}{block}.pickle"
    else:
        d_file = f"data/gevp_{nn_iso}_{tnorm}_gevp_{gevp_plot}{block}.pickle"
    nn_data = gv.load(d_file)

    plt.ion()
    tf_results = {}
    tf_lbls    = []
    for q in states:
        start_time = time.perf_counter()
        tf_results['_'.join([str(k) for k in q])] = {'DE':[],'E1':[],'E2':[]}
        print(q)
        q_str = '\_'.join([str(k) for k in q])
        fig = plt.figure(str(q),figsize=(7,5.5))
        ax_nn  = plt.axes([0.15, 0.66, 0.84, 0.33])
        ax_nnR = plt.axes([0.15, 0.33, 0.84, 0.33])
        ax_Q   = plt.axes([0.15, 0.13, 0.84, 0.20])

        # plot fit on data
        params_q = dict(optimal_p)
        params_q['ratio'] = args.ratio
        for k in post_optimal:
            if k[0][0] == q:
                #print(k)#,post_optimal[k])
                params_q[k] = post_optimal[k]
                if k[1] == 'e0' and k[0][1] == 'R':
                    e0_opt = post_optimal[k]
                    k_opt  = k[0]
                    k_n    = k[0][2]
                    k_n1   = (k[0][0],"N",k[0][2][0])
                    k_n2   = (k[0][0],"N",k[0][2][1])
                    e1_opt = post_optimal[(k_n1, "e0")]
                    e2_opt = post_optimal[(k_n2, "e0")]
        # plot fit on numerator
        fit_func = fitter.Functions(params_q)
        x_plot = np.arange(0,20,.1)
        nn_opt = fit_func.pure_ratio(k_opt, x_plot, params_q)
        eff_opt = np.log(nn_opt / np.roll(nn_opt,-10))
        y  = np.array([eff.mean for eff in eff_opt])
        dy = np.array([eff.sdev for eff in eff_opt])
        ax_nn.axvspan(0,opt_tmin-0.5,color='k',alpha=.2)
        ax_nn.axvspan(15.5,20,color='k',alpha=.2)
        ax_nn.fill_between(x_plot,y-dy, y+dy, color=opt_clr,alpha=.3)
        # plot g.s. e0
        e_nn = e0_opt +e1_opt +e2_opt
        ax_nn.axhline(e_nn.mean-e_nn.sdev, linestyle='--',color=opt_clr, alpha=.3)
        ax_nn.axhline(e_nn.mean+e_nn.sdev, linestyle='--',color=opt_clr, alpha=.3)

        # plot fit on ratio
        n1_opt = fit_func.twopoint(k_n1, x_plot, params_q, "N")
        n2_opt = fit_func.twopoint(k_n2, x_plot, params_q, "N")
        r_opt  = nn_opt / n1_opt / n2_opt
        eff_opt = np.log(r_opt / np.roll(r_opt,-10))
        y  = np.array([eff.mean for eff in eff_opt])
        dy = np.array([eff.sdev for eff in eff_opt])
        ax_nnR.axvspan(0,opt_tmin-0.5,color='k',alpha=.2)
        ax_nnR.axvspan(15.5,20,color='k',alpha=.2)
        ax_nnR.fill_between(x_plot,y-dy, y+dy, color=opt_clr,alpha=.3)
        # plot g.s. e0
        ax_nnR.axhline(e0_opt.mean-e0_opt.sdev, linestyle='--',color=opt_clr, alpha=.3)
        ax_nnR.axhline(e0_opt.mean+e0_opt.sdev, linestyle='--',color=opt_clr, alpha=.3)
        

        # plot e0 from stability
        plot_tmin(ax_nn, ax_nnR, ax_Q, q, models, args, nn_file, nn_dict, nn_model, optimal_model, fit_keys, nn_data, tf_results, tf_lbls)

        fig_name = '%s_tf_NN_%s' %(q_str.replace('\_','_'), args.optimal.split('/')[-1].replace('pickle','stability.pdf'))
        plt.savefig('figures/'+fig_name,transparent=True)

        stop_time = time.perf_counter()
        print('\n%.0f seconds' %(stop_time - start_time))

    # make summary plot
    try:
        mN = tf_results['0_T1g_0']['E1'][0]
    except:
        mN = tf_results['0_A1g_0']['E1'][0]

    # plot t0_N
    summary_plot.summary_ENN(tf_results, mN, tf_lbls, color, 
                             lbl0=r'$t_f^{NN}$=', spin=nn_iso, fig=f"{nn_iso}_tfNN_summary")

    plt.ioff()
    plt.show()

def plot_tmin(axnn, axnnR, axQ, state, models, arg, nnFile, nnDict, nnModel, optModel, fitKeys, nnData, r_tf, l_tf, ratio=True):

    q_str = '\_'.join([str(k) for k in state])

    for t in arg.tf_NN:
        plot_one_tmin(t, axnn, axnnR, axQ, state, models, arg, nnFile, nnDict, nnModel, optModel, fitKeys, nnData, r_tf, l_tf)

    k_n     = fitKeys[state][0][2]
    nn_corr = nnData[state][2:]
    n1_corr = nnData[k_n[0]][2:]
    n2_corr = nnData[k_n[1]][2:]
    r_corr  = nn_corr / n1_corr / n2_corr

    nn_eff  = np.log(nn_corr/np.roll(nn_corr,-1))
    r_eff   = np.log(r_corr/np.roll(r_corr,-1))

    m  = np.array([k.mean for k in nn_eff])
    dm = np.array([k.sdev for k in nn_eff])
    axnn.errorbar(np.arange(2,2+len(nn_eff),1),m,yerr=dm,color='k',mfc='None',marker='o',linestyle='None', label=r'eff mass')

    m  = np.array([k.mean for k in r_eff])
    dm = np.array([k.sdev for k in r_eff])
    axnnR.errorbar(np.arange(2,2+len(r_eff),1),m,yerr=dm,color='k',mfc='None',marker='o',linestyle='None', label=r'eff mass')

    handles, labels = axnnR.get_legend_handles_labels()
    axnn.legend(flip(handles, len(arg.tf_NN)), flip(labels, len(arg.tf_NN)), loc=1, ncol=len(arg.tf_NN), fontsize=10, columnspacing=0,handletextpad=0.1)

    nnr_lim = summary_plot.nnr_lim
    nn_lim  = summary_plot.nn_lim

    axnnR.set_ylim(nnr_lim[state])
    axnn.set_ylim(nn_lim[state])
    axnnR.set_ylabel(r'$\Delta E_0^{\rm %s}$' %q_str, fontsize=20)
    axnn.set_ylabel(r'$E_0^{\rm %s}$' %q_str, fontsize=20)
    axQ.set_ylabel(r'$Q$', fontsize=20)
    axQ.tick_params(bottom=True, top=True, direction='in')
    axQ.set_xlabel(r'$t_{\rm max}^{NN}$', fontsize=20)

    axnn.set_xlim(1.5,17.5)
    axnnR.set_xlim(1.5,17.5)
    axQ.set_xlim(1.5, 17.5)

    q_up = 1.05
    axQ.set_ylim(0,q_up)

    axnn.tick_params(direction='inout')
    axnn.set_xticklabels([])
    axnnR.tick_params(direction='inout')
    axnnR.set_xticklabels([])
    axQ.tick_params(direction='inout')
    #axQ.set_xticklabels([])
    axQ.set_yticks([0, .25, .5, .75])


def plot_one_tmin(t, axnn, axnnR, axQ, state, models, arg, nnFile, nnDict, nnModel, optModel, fitKeys, nnData, r_tf, l_tf):
    # t = tNN_f
    marker = {
        'N_n4_NN_conspire_e0':'s',
        'N_n3_NN_conspire_e0':'*',
        'N_n2_NN_conspire_e0':'o',
    }
    shift = { 10:-0.2, 11:-0.1, 12:0., 13:0.1, 14:0.2, 15:0.3 }
    color = { 10:'orange', 11:'r', 12:'g', 13:'b', 14:'magenta', 15:'gray' }


    opt_tmin = int(arg.optimal.split('_NN_')[1].split('-')[0].split('_')[-1])

    e      = []
    e_nn   = []
    p      = []
    m_plot = []
    c_plot = []
    t_plot = []
    mfc_plot = []
    for model in models:
        for t0 in arg.tmin:
            # don't track correlated gvars between all gv.load calls
            gv.switch_gvar()

            sys.stdout.write('  tf^NN=%d, %s, t0_NN=%4s\r' %(t, model, t0))
            sys.stdout.flush()
            fit_model = nnModel.format(**models[model])
            n_inel    = models[model]['N_inel']
            nn_el     = models[model]['nn_el']
            if 'agnostic' in fit_model:
                nn    = int(fit_model.split('agnostic_n')[1].split('_')[0])
            nnDict.update({'N_inel':n_inel, 'nn_el':nn_el, 't0':t0, 'tf':t,
                           'nn_model':models[model]['nn_model']})

            if t0 == arg.tmin[0]:
                if t == arg.tf_NN[0]:
                    lbl = r'$N_{\rm inel} = %d, t_f^{NN} = %d$' %(n_inel, t)
                else:
                    lbl = r'$%d, %d$' %(n_inel, t)
            else:
                lbl = ''

            fit_file = 'result/'+nnFile.format(**nnDict)
            if os.path.exists(fit_file):
                if arg.debug:
                    print('\nDEBUG: fit file', fit_file)
                    print('DEBUG: t0 = ',t0)
                    print('DEBUG: tf = ',t)
                    print('DEBUG: nn_file', nnFile)

                data = gv.load(fit_file)
                #if arg.optimal:
                try:
                    e.append(data[fitKeys[state]])
                except:
                    print(fit_file)
                    sys.exit()
                p.append(data[((state,), 'Q')])
                k_n = fitKeys[state][0][2]
                k_tmp  = fitKeys[state]
                k_n1   = (k_tmp[0][0],"N",k_tmp[0][2][0])
                k_n2   = (k_tmp[0][0],"N",k_tmp[0][2][1])
                e1_opt = data[(k_n1, "e0")]
                e2_opt = data[(k_n2, "e0")]
                e_nn.append(e[-1] + e1_opt + e2_opt)
                #else:
                #    for k in data:
                #        if k[1] == 'e0' and k[0][1] == 'R' and k[0][0] == state:
                #            e.append(data[(k[0], 'e0')])
                #            p.append(data[((state,), 'Q')])
                #            k_n = fitKeys[state][0][2]
                mrkr = marker[fit_model]
                clr  = color[t]
                m_plot.append(marker[fit_model])
                c_plot.append(color[t])
                t_plot.append(t0+shift[t])
                mfc='None'
                if t == opt_tmin and fit_model == optModel and t0==opt_tmin:
                    mfc='k'#clr
                mfc_plot.append(mfc)
                axnnR.errorbar(t0+shift[t],
                                e[-1].mean, yerr=e[-1].sdev,
                                marker=mrkr, color=clr, mfc=mfc,
                                linestyle='None',label=lbl)
                axnn.errorbar(t0+shift[t],
                                e_nn[-1].mean, yerr=e_nn[-1].sdev,
                                marker=mrkr, color=clr, mfc=mfc,
                                linestyle='None',label=lbl)
                # populate results for comparison
                # t0 == tNN_0
                # t  == tNN_f
                if t0 == opt_tmin and fit_model == optModel:
                    l_tf.append(t)
                    r_tf['_'.join([str(k) for k in state])]['DE'].append(e[-1])
                    r_tf['_'.join([str(k) for k in state])]['E1'].append(e1_opt)
                    r_tf['_'.join([str(k) for k in state])]['E2'].append(e2_opt)
            else:
                print('missing', fit_file)

            # delete gvars from memory
            gv.restore_gvar()

    e = np.array(e)
    p = np.array(p)
    mfc='None'
    for i_p,pp in enumerate(p):
        axQ.plot(t_plot[i_p], p[i_p], linestyle='None',
                    marker=m_plot[i_p], color=c_plot[i_p], mfc=mfc_plot[i_p],)

    if t == arg.tf_NN[0]:
        p_plot = np.array(p)





if __name__ == "__main__":
    main()
