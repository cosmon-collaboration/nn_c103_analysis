import numpy as np
import gvar as gv
import os, sys
import h5py as h5

BMat_path='/Users/walkloud/work/research/c51/x_files/code/pythib'
sys.path.append(BMat_path)
import BMat

L=48
irreps_clrs = {
    'T1g':'k', 'A2':'b', 'E':'r', 'B1':'g', 'B2':'magenta',
    'A1g':'k', 'A1':'b',
}

level_mrkr = {0:'s', 1:'o', 2:'d', 3:'p', 4:'h', 5:'8', 6:'v'}

momRay = {0 : 'ar', 1 : 'oa', 2 : 'pd', 3 : 'cd', 4 : 'oa' }
def calcFunc(self, JtimesTwo, Lp, SptimesTwo, chanp, L, StimesTwo, chan, Ecm_over_mref, pSqFuncList):
    return 0.
chanList = [BMat.DecayChannelInfo('n','n',1,1,True,True),]


def get_data(states, nn_type,path,bs_bias_correct=True,Mbs=None, irrep_avg=[], PsqMax=4):
    if any([k in nn_type for k in ['deuteron', 'result']]):
        def isZero(JtimesTwo, Lp, SptimesTwo, chanp, L, StimesTwo, chan):
            return not (JtimesTwo==2
                    and Lp==0 and L==0
                    and chanp==0 and chan==0
                    and SptimesTwo==2 and StimesTwo==2)
    elif nn_type == 'dineutron':
        def isZero(JtimesTwo, Lp, SptimesTwo, chanp, L, StimesTwo, chan):
            return not (JtimesTwo==0
                    and Lp==0 and L==0
                    and chanp==0 and chan==0
                    and SptimesTwo==0 and StimesTwo==0)
    Kinv = BMat.KMatrix(calcFunc, isZero)
    fit_results = gv.load(path)
    try:
        mN = np.array(fit_results[((('0', 'T1g', 0), 'N', '0'), 'e0')])
    except:
        mN = np.array(fit_results[((('0', 'A1g', 0), 'N', '0'), 'e0')])

    n_bs = len(mN[1:])
    if Mbs is not None:
        if Mbs > n_bs:
            sys.exit('you only have '+str(n_bs)+' bs total samples')
        Nbs = Mbs
        mN = mN[:Nbs+1]
    else:
        Nbs = n_bs

    if irrep_avg:
        '''
        for k in irrep_avg:
            print(k)
            for w,sub in irrep_avg[k]:
                print('   +',w,sub)
        '''
        irrep_grps = irrep_avg
    else:
        irrep_grps = {}
        for k in states:
            irrep_grps[k] = [(1, k)]

    results = {}
    results['energies_0']  = {}
    results['energies_bs'] = {}
    results['qcotd_0']     = {}
    results['qcotd_bs']    = {}
    results['qsq_0']       = {}
    results['qsq_bs']      = {}
    results['en1']         = {}
    results['en2']         = {}

    results['energies_0']['m_n']  = np.array(mN[0])
    results['energies_bs']['m_n'] = np.array(mN[1:])

    excluded = []

    if os.path.exists('conspiracy_c103_results.h5'):
        os.remove('conspiracy_c103_results.h5')

    for grp in irrep_grps:
        de_nn = 0
        en1   = 0
        en2   = 0
        for w, state in irrep_grps[grp]:
            for k in fit_results:
                if len(k[0]) == 3 and len(k[0][0]) == 3 and (k[0][0] == state and k[0][1] == 'R' and k[1] == 'e0'):
                    grp_full = ((grp, 'R', k[0][2]), 'e0')

                    Psq, irrep, n = k[0][0]
                    Psq = int(Psq)
                    de_nn += w * np.array(fit_results[k])[:Nbs+1]
                    s1,s2  = k[0][2]
                    st1    = ((k[0][0], 'N', s1), 'e0')
                    st2    = ((k[0][0], 'N', s2), 'e0')
                    en1   += w * np.array(fit_results[st1])[:Nbs+1]
                    en2   += w * np.array(fit_results[st2])[:Nbs+1]
        e_nn   = de_nn + en1 + en2
        E_cmSq = e_nn**2 - Psq*(2*np.pi/L)**2
        qsq    = E_cmSq / 4 - mN**2

        # make mean (boot0) values
        mN_0   = mN[0]
        e_nn_0 = e_nn[0]
        irrep  = irrep_grps[grp][0][1][1]
        boxQ   = BMat.BoxQuantization(momRay[Psq], Psq, irrep, chanList, [0,], Kinv, True)
        boxQ.setRefMassL(mN[0]*L)
        boxQ.setMassesOverRef(0, 1, 1)
        qcotd  = boxQ.getBoxMatrixFromElab(e_nn_0 / mN_0)# in mN units

        if len(irrep_grps[grp]) > 1:
            irrep2 = irrep_grps[grp][1][1][1]
            box2   = BMat.BoxQuantization(momRay[Psq], Psq, irrep2, chanList, [0,], Kinv, True)
            box2.setRefMassL(mN[0]*L)
            box2.setMassesOverRef(0, 1, 1)
            qcotd2  = box2.getBoxMatrixFromElab(e_nn_0 / mN_0)# in mN units
            #import IPython; IPython.embed()

        # make bootstrap distribution
        qsq_qcotd_bs = np.zeros([Nbs,2])
        boxQ_bs      = BMat.BoxQuantization(momRay[Psq], Psq, irrep, chanList, [0,], Kinv, True)
        boxQ_bs.setMassesOverRef(0,1,1)

        mN_bs     = mN[1:]
        e_nn_bs   = e_nn[1:]
        if bs_bias_correct:
            dmN     = mN_bs - mN_bs.mean()
            mN_bs   = mN_0 + dmN
            denn    = e_nn_bs - e_nn_bs.mean()
            e_nn_bs = e_nn_0 + denn

        E_cmSq_bs = e_nn_bs**2 - Psq*(2*np.pi/L)**2
        qsq_qcotd_bs[:,0] = E_cmSq_bs/4 - mN_bs**2
        for bs in range(Nbs):
            boxQ_bs.setRefMassL(mN_bs[bs]*L)
            qsq_qcotd_bs[bs,1] = boxQ_bs.getBoxMatrixFromElab(e_nn_bs[bs] / mN_bs[bs]).real

        # save data for ERE analysis
        gv_en1   = gv.gvar(en1[0],             en1[1:].std())
        gv_en2   = gv.gvar(en2[0],             en2[1:].std())
        gv_de_nn = gv.gvar(de_nn[0],           de_nn[1:].std())
        gv_e_nn  = gv.gvar(e_nn[0],            e_nn[1:].std())
        gv_E_cm  = gv.gvar(np.sqrt(E_cmSq[0]), np.sqrt(E_cmSq[1:]).std())
        gv_qsq   = gv.gvar(qsq[0]/mN[0]**2,    (qsq[1:]/mN_bs**2).std())
        gv_qcotd = gv.gvar(qcotd.real, qsq_qcotd_bs[:,1].std())

        if qsq[0]/mN_0**2 < 0.05 and np.real(qcotd) < 10 and np.real(qcotd) > -0.2 and Psq <= PsqMax:
            results['energies_0'][grp_full] = e_nn[0]
            results['qcotd_0'][grp_full]    = qcotd.real
            results['qsq_0'][grp_full]      = qsq[0]/mN[0]**2
            #print('%d& %3s& %s& %s& %s& %s& %s& %s& %s& %s& %s& %s\\\\' \
            #    %(Psq, irrep, n, s1, gv_en1, s2, gv_en2, gv_de_nn, gv_e_nn, gv_E_cm, gv_qsq, gv_qcotd))

            results['energies_bs'][grp_full] = e_nn_bs
            results['qcotd_bs'][grp_full]    = qsq_qcotd_bs[:,1]
            results['qsq_bs'][grp_full]      = qsq_qcotd_bs[:,0] / mN[1:]**2

            results['en1'][grp_full]         = en1
            results['en2'][grp_full]         = en2

            with h5.File('conspiracy_c103_results.h5','a') as f5:
                group = '%s_%s_Psq%d' %(grp[1], n, Psq)
                f5.create_dataset(group+'/dEnn', data = de_nn)
                f5.create_dataset(group+'/N1', data   = en1)
                f5.create_dataset(group+'/N2', data   = en2)
                f5.create_dataset(group+'/mN', data   = mN)

        else:
            excluded.append('%d& %3s& %s& %s& %s& %s& %s& %s& %s& %s& %s& %s\\\\' \
                %(Psq, irrep, n, s1, gv_en1, s2, gv_en2, gv_de_nn, gv_e_nn, gv_E_cm, gv_qsq, gv_qcotd))

    print('\nExcluded')
    if len(excluded) == 0:
        print('None')
    else:
        for e in excluded:
            print(e)

    return results
