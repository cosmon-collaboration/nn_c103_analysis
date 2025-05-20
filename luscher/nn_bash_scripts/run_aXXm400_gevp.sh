#!/bin/bash

#ensemble='a15m400trMc'
ensemble='a12m400trMc'

if [[ $ensemble == "a15m400trMc" ]]; then
    gevp="3-6 3-7 3-8 4-7 4-8 4-9"
    block=2
    tNN=`seq 2 8`
    tf=10
    t0N="2 3 4 5 6"
    tf_N=15
    ref_time=7
    mpi=0.31668
    mN=0.8678
elif [[ $ensemble == "a12m400trMc" ]]; then
    gevp="4-7 4-8 4-9 4-10 5-9 5-10 5-11"
    block=2
    tNN=`seq 2 10`
    tf=13
    t0N="2 3 4 5 6"
    tf_N=18
    ref_time=10
    mpi=0.25357
    mN=0.7045
fi

ratio="False"
nn_iso='triplet'
tnorm=3
e=0

for gevp_t in $gevp; do
    t0=${gevp_t%-*}
    td=${gevp_t#*-}
    gevp="${t0}-${td}"
    echo "$t0, $td, $gevp_t, $gevp"
    for n_N in 2 3 4; do
        for t0_N in $t0N; do
            nucleon="n${n_N}_t_${t0_N}-${tf_N}"
            for t in $tNN; do
                result="result_${ensemble}/NN_${nn_iso}_tnorm${tnorm}_t0-td_${gevp}_N_${nucleon}_NN_conspire_e${e}_t_${t}-${tf}_ratio_${ratio}_block${block}.pickle"
                echo ""
                echo $result
                sed "s/aXXm400trMc/${ensemble}/" h-dibaryon_aXXm400trMc_base.py \
                | sed "s/t0\"\] = 4/t0\"\] = ${t0}/" \
                | sed "s/td\"\] = 8/td\"\] = ${td}/" \
                | sed "s/t_norm\'\] = 3/t_norm\'\] = ${tnorm}/" \
                | sed "s/block\"\] = 2/block\"\] = $block/" \
                | sed "s/nstates\"]     = 3/nstates\"]     = ${n_N}/" \
                | sed "s/N\": \[3, 15\]/N\": \[${t0_N}, ${tf_N}\]/" \
                | sed "s/R\": \[3, 12\]/R\": \[$t, $tf\]/" \
                | sed "s/ratio\"]       = True/ratio\"]       = ${ratio}/" \
                | sed "s/ref_time/${ref_time}/" \
                | sed "s/a_mpi/${mpi}/" \
                | sed "s/a_mn/${mN}/" \
                > h-dibaryon_${ensemble}.py
                if [[ ! -e $result ]]; then
                    ln -sf h-dibaryon_${ensemble}.py nn_parameters.py
                    python nn_fit.py
                else
                    echo "  already fit"
                fi
            done
        done
    done
done
