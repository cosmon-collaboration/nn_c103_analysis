#!/usr/bin/env python3
import os, sys, time
import gvar as gv
import numpy as np
import matplotlib.pyplot as plt
# load nn_fit to get fit functions
import nn_fit as fitter
import argparse
import itertools
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

def main():
    parser = argparse.ArgumentParser(
        description='Plot results from NN fits with various correlator models')
    # optimal fit 
    parser.add_argument('optimal',  type=str, 
                        help=       'add optimal fit result')

    # NN info
    parser.add_argument('--nn_iso', type=str, default='singlet',
                        help=       'NN system: singlet or triplet [%(default)s]')
    parser.add_argument('--n_N',    nargs='+', type=int, default=[3,4],
                        help=       'number of exponentials in single nucleon to sweep over %(default)s')
    parser.add_argument('--nn_el',  nargs='+', default=[0],
                        help=       'number of elastic e.s. to try %(default)s')
    parser.add_argument('--ratio',  default=False, action='store_true',
                        help=       'fit from RATIO correlator? [%(default)s]')
    parser.add_argument('--gevp',   nargs='+', type=str, default=['3-8', '3-10','4-8', '4-10','5-10','6-10'],
                        help=       'list of GEVP times in t0-td format %(default)s')
    parser.add_argument('--tmin',   nargs='+', default=range(2,11),
                        help=       'values of t_min in NN fit [%(default)s]')
   
    args = parser.parse_args()
    print(args)

    color = { '3-10':'magenta', '4-10':'b', '5-10':'g', '6-10':'r', '3-8':'orange', '4-8':'yellow' }

    N_t = args.optimal.split('_NN')[0].split('_')[-1]

    nn_file  = 'NN_{nn_iso}_t0-td_{gevp}_N_n{N_inel}_t_{N_t}'
    nn_file += '_NN_{nn_model}_e{nn_el}_t_{t0}-15_ratio_'+str(args.ratio)+'.pickle'
    nn_dict = { 'N_t':N_t, 'nn_iso':args.nn_iso, }

    nn_model = 'N_n{N_inel}_NN_{nn_model}_e{nn_el}'

    models = {}
    for n in args.n_N:
        models['N_n%d_NN_conspire_e0' %n] = {'N_inel':n, 'nn_model':'conspire', 'nn_el':0}
    
    if not os.path.exists("figures"):
        os.mkdir("figures")

    states = [
        ('0', 'T1g', 0), ('0', 'T1g', 1), ('1', 'A2', 0), ('1', 'A2', 1),
        ('2', 'A2', 0),  ('3', 'A2', 0),  ('4', 'A2', 0), ('4', 'A2', 1),
        ('2', 'B1', 0),  ('2', 'B2', 0),  ('2', 'B2', 3), ('1', 'E', 0),
        ('1', 'E', 1),   ('3', 'E', 0),   ('4', 'E', 0),  ('4', 'E', 1)
    ]
    #states = [('0', 'T1g', 0)]
    #states = [('3', 'A2', 0)]

    if args.optimal:
        print('loading optimal fit:',args.optimal)
        post_optimal  = gv.load(args.optimal)
        optimal_p     = {'positive_z':True, 'debug':False}
        N_inel        = int(args.optimal.split('N_n')[1].split('_')[0])
        nn_el         = int(args.optimal.split('_e')[1].split('_')[0])
        optimal_model = {}
        if 'conspire' in args.optimal:
            optimal_p['version']      = 'conspire'
            optimal_model['nn_model'] = 'conspire'
        else:
            optimal_p['version'] = 'agnostic'
            r_n_inel = int(args.optimal.split('agnostic_n')[1].split('_')[0])
            optimal_p['r_n_inel']  = r_n_inel
            optimal_model['model'] = 'agnostic_n%d' %r_n_inel
        
        optimal_p['r_n_el']  = nn_el
        optimal_p['nstates'] = N_inel
        optimal_model['N_inel'] = N_inel
        optimal_model['nn_el']  = nn_el

        optimal_model = nn_model.format(**optimal_model)

        color = { '3-10':'magenta', '4-10':'b', '5-10':'g', '6-10':'r', '3-8':'orange', '4-8':'yellow' }
        opt_tmin  = int(args.optimal.split('_NN_')[1].split('-')[0].split('_')[-1])
        gevp_plot = args.optimal.split('t0-td_')[1].split('_')[0]
        opt_clr   = color[gevp_plot]

        # get data keys
        fit_keys = {}
        for q in states:
            for k in post_optimal:
                if k[1] == 'e0' and k[0][1] == 'R' and k[0][0] == q:
                    fit_keys[q] = k

    else:
        gevp_plot = args.gevp[0]

    nn_data = gv.load('data/gevp_'+args.nn_iso+'_'+gevp_plot+'.pickle')

    plt.ion()
    for q in states:
        start_time = time.perf_counter()
        print(q)
        q_str = '\_'.join([str(k) for k in q])
        fig = plt.figure(str(q),figsize=(7,5.5))
        ax_nn  = plt.axes([0.15, 0.66, 0.84, 0.33])
        ax_nnR = plt.axes([0.15, 0.33, 0.84, 0.33])
        ax_Q   = plt.axes([0.15, 0.13, 0.84, 0.20])

        if args.optimal:
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
        plot_tmin(ax_nn, ax_nnR, ax_Q, q, models, args, nn_file, nn_dict, nn_model, optimal_model, fit_keys, nn_data)

        if args.optimal:
            fig_name = '%s_%s' %(q_str.replace('\_','_'), args.optimal.split('/')[-1].replace('pickle','stability.pdf'))
        else:
            fig_name = q_str.replace('\_','_')+'_stability.pdf'
        plt.savefig('figures/'+fig_name,transparent=True)

        stop_time = time.perf_counter()
        print('\n%.0f seconds' %(stop_time - start_time))


    plt.ioff()
    plt.show()

def plot_tmin(axnn, axnnR, axQ, state, models, arg, nnFile, nnDict, nnModel, optModel, fitKeys, nnData, ratio=True):

    q_str = '\_'.join([str(k) for k in state])

    for t in arg.tmin:
        plot_one_tmin(t, axnn, axnnR, axQ, state, models, arg, nnFile, nnDict, nnModel, optModel, fitKeys, nnData)
        '''
        e      = []
        e_nn   = []
        p      = []
        m_plot = []
        c_plot = []
        t_plot = []
        mfc_plot = []
        for model in models:
            for gevp in arg.gevp:
                # don't track correlated gvars between all gv.load calls
                gv.switch_gvar()

                sys.stdout.write('  t=%d, %s, %4s\r' %(t, model, gevp))
                sys.stdout.flush()
                fit_model = nnModel.format(**models[model])
                n_inel    = models[model]['N_inel']
                nn_el     = models[model]['nn_el']
                if 'agnostic' in fit_model:
                    nn    = int(fit_model.split('agnostic_n')[1].split('_')[0])
                nnDict.update({'gevp':gevp, 'N_inel':n_inel, 'nn_el':nn_el, 't0':t, 
                               'nn_model':models[model]['nn_model']})

                if t == arg.tmin[0]:
                    if gevp == arg.gevp[0]:
                        lbl = r'$N_{\rm inel} = %d, t_0-t_d = %s$' %(n_inel, gevp)
                    else:
                        lbl = r'$%d, %s$' %(n_inel, gevp)
                else:
                    lbl = ''

                fit_file = 'result/'+nnFile.format(**nnDict)
                if os.path.exists(fit_file):

                    data = gv.load(fit_file)
                    if arg.optimal:
                        e.append(data[fitKeys[state]])
                        p.append(data[((state,), 'Q')])
                        k_n = fitKeys[state][0][2]
                        k_tmp  = fitKeys[state]
                        k_n1   = (k_tmp[0][0],"N",k_tmp[0][2][0])
                        k_n2   = (k_tmp[0][0],"N",k_tmp[0][2][1])
                        e1_opt = data[(k_n1, "e0")]
                        e2_opt = data[(k_n2, "e0")]
                        e_nn.append(e[-1] + e1_opt + e2_opt)
                    else:
                        for k in data:
                            if k[1] == 'e0' and k[0][1] == 'R' and k[0][0] == state:
                                e.append(data[(k[0], 'e0')])
                                p.append(data[((state,), 'Q')])
                                k_n = fitKeys[state][0][2]
                    mrkr = marker[fit_model]
                    clr  = color[gevp]
                    m_plot.append(marker[fit_model])
                    c_plot.append(color[gevp])
                    t_plot.append(t+shift[gevp])
                    mfc='None'
                    if arg.optimal and t == opt_tmin and fit_model == optModel and gevp==arg.gevp[0]:
                        mfc='k'#clr
                    mfc_plot.append(mfc)
                    axnnR.errorbar(t+shift[gevp],
                                  e[-1].mean, yerr=e[-1].sdev,
                                  marker=mrkr, color=clr, mfc=mfc,
                                  linestyle='None',label=lbl)
                    axnn.errorbar(t+shift[gevp],
                                  e_nn[-1].mean, yerr=e_nn[-1].sdev,
                                  marker=mrkr, color=clr, mfc=mfc,
                                  linestyle='None',label=lbl)
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

        if t == arg.tmin[0]:
            p_plot = np.array(p)
        '''

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
    axnn.legend(flip(handles, len(arg.gevp)), flip(labels, len(arg.gevp)), loc=1, ncol=len(arg.gevp), fontsize=10, columnspacing=0,handletextpad=0.1)

    nnr_lim = {
        ('0', 'T1g', 0):(-0.0026,0.0009), ('0', 'T1g', 1):(-0.0051,0.0009),
        ('1', 'A2', 0) :(-0.0051,0.0009), ('1', 'A2', 1) :(-0.0051,0.0009),
        ('2', 'A2', 0) :(-0.0031,0.0009), ('3', 'A2', 0) :(-0.0031,0.0009),
        ('4', 'A2', 0) :(-0.0021,0.0009), ('4', 'A2', 1) :(-0.0031,0.0009),
        ('2', 'B1', 0) :(-0.0031,0.0009), ('2', 'B2', 0) :(-0.0031,0.0009),
        ('2', 'B2', 3) :(-0.0031,0.0009), ('1', 'E', 0)  :(-0.0031,0.0009),
        ('1', 'E', 1)  :(-0.0051,0.0009), ('3', 'E', 0)  :(-0.0051,0.0009),
        ('4', 'E', 0)  :(-0.0031,0.0009), ('4', 'E', 1)  :(-0.0031,0.0009)
    }
    nn_lim = {
        ('0', 'T1g', 0):(1.400,1.445), ('0', 'T1g', 1):(1.420,1.465),
        ('1', 'A2', 0) :(1.405,1.450), ('1', 'A2', 1) :(1.430,1.475),
        ('2', 'A2', 0) :(1.420,1.465), ('3', 'A2', 0) :(1.430,1.475),
        ('4', 'A2', 0) :(1.420,1.465), ('4', 'A2', 1) :(1.445,1.490),
        ('2', 'B1', 0) :(1.420,1.465), ('2', 'B2', 0) :(1.425,1.470),
        ('2', 'B2', 3) :(1.445,1.490), ('1', 'E', 0)  :(1.410,1.465),
        ('1', 'E', 1)  :(1.430,1.475), ('3', 'E', 0)  :(1.435,1.480),
        ('4', 'E', 0)  :(1.425,1.470), ('4', 'E', 1)  :(1.445,1.490)
    }

    axnnR.set_ylim(nnr_lim[state])
    axnn.set_ylim(nn_lim[state])
    axnnR.set_ylabel(r'$\Delta E_0^{\rm %s}$' %q_str, fontsize=20)
    axnn.set_ylabel(r'$E_0^{\rm %s}$' %q_str, fontsize=20)
    axQ.set_ylabel(r'$Q$', fontsize=20)
    axQ.tick_params(bottom=True, top=True, direction='in')
    axQ.set_xlabel(r'$t_{\rm min}$', fontsize=20)

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


def plot_one_tmin(t, axnn, axnnR, axQ, state, models, arg, nnFile, nnDict, nnModel, optModel, fitKeys, nnData):
    marker = {
        'N_n4_NN_conspire_e0':'s',
        'N_n3_NN_conspire_e0':'*',
        'N_n2_NN_conspire_e0':'o',
    }
    shift = { '3-10':0., '4-10':0.1, '5-10':0.2, '6-10':0.3, '3-8':-0.2, '4-8':-0.1 }
    color = { '3-10':'magenta', '4-10':'b', '5-10':'g', '6-10':'r', '3-8':'orange', '4-8':'yellow' }

    if arg.optimal:
        opt_tmin = int(arg.optimal.split('_NN_')[1].split('-')[0].split('_')[-1])

    e      = []
    e_nn   = []
    p      = []
    m_plot = []
    c_plot = []
    t_plot = []
    mfc_plot = []
    for model in models:
        for gevp in arg.gevp:
            # don't track correlated gvars between all gv.load calls
            gv.switch_gvar()

            sys.stdout.write('  t=%d, %s, %4s\r' %(t, model, gevp))
            sys.stdout.flush()
            fit_model = nnModel.format(**models[model])
            n_inel    = models[model]['N_inel']
            nn_el     = models[model]['nn_el']
            if 'agnostic' in fit_model:
                nn    = int(fit_model.split('agnostic_n')[1].split('_')[0])
            nnDict.update({'gevp':gevp, 'N_inel':n_inel, 'nn_el':nn_el, 't0':t, 
                            'nn_model':models[model]['nn_model']})

            if t == arg.tmin[0]:
                if gevp == arg.gevp[0]:
                    lbl = r'$N_{\rm inel} = %d, t_0-t_d = %s$' %(n_inel, gevp)
                else:
                    lbl = r'$%d, %s$' %(n_inel, gevp)
            else:
                lbl = ''

            fit_file = 'result/'+nnFile.format(**nnDict)
            if os.path.exists(fit_file):

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
                clr  = color[gevp]
                m_plot.append(marker[fit_model])
                c_plot.append(color[gevp])
                t_plot.append(t+shift[gevp])
                mfc='None'
                if t == opt_tmin and fit_model == optModel and gevp==arg.gevp[0]:
                    mfc='k'#clr
                mfc_plot.append(mfc)
                axnnR.errorbar(t+shift[gevp],
                                e[-1].mean, yerr=e[-1].sdev,
                                marker=mrkr, color=clr, mfc=mfc,
                                linestyle='None',label=lbl)
                axnn.errorbar(t+shift[gevp],
                                e_nn[-1].mean, yerr=e_nn[-1].sdev,
                                marker=mrkr, color=clr, mfc=mfc,
                                linestyle='None',label=lbl)
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

    if t == arg.tmin[0]:
        p_plot = np.array(p)





if __name__ == "__main__":
    main()