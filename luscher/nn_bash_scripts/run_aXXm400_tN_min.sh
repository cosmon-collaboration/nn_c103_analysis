#!/bin/bash

ensemble='a15m400trMc'
gevp_t="3-7"
n_N="2"
Nb=2
tNN=`seq 2 8`
tf=10
t0N="2 3 4 5 6"
tf_N=15

ratio="False"
nn_iso='triplet'
tnorm=3
e=0

for gevp_t in $gevp_t; do
    t0=${gevp_t%-*}
    td=${gevp_t#*-}
    gevp="${t0}-${td}"
    echo "$t0, $td, $gevp_t, $gevp"
    for block in $Nb; do
        for t0_N in $t0N; do
            nucleon="n${n_N}_t_${t0_N}-$tf_N"
            for t in $tNN; do
                if [[ $block -eq 1 ]]; then
                    result="result_${ensemble}/NN_${nn_iso}_tnorm${tnorm}_t0-td_${gevp}_N_${nucleon}_NN_conspire_e${e}_t_${t}-${tf}_ratio_${ratio}.pickle"
                else
                    result="result_${ensemble}/NN_${nn_iso}_tnorm${tnorm}_t0-td_${gevp}_N_${nucleon}_NN_conspire_e${e}_t_${t}-${tf}_ratio_${ratio}_block${block}.pickle"
                fi
                echo ""
                echo $result
                if [[ ! -e $result ]]; then
                    sed "s/aXXm400trMc/${ensemble}/" h-dibaryon_aXXm400trMc_base.py \
                    | sed "s/t0\"\] = 4/t0\"\] = ${t0}/" \
                    | sed "s/td\"\] = 8/td\"\] = ${td}/" \
                    | sed "s/t_norm\'\] = 3/t_norm\'\] = ${tnorm}/" \
                    | sed "s/block\"\] = 2/block\"\] = $block/" \
                    | sed "s/nstates\"]     = 3/nstates\"]     = ${n_N}/" \
                    | sed "s/N\": \[3, 15\]/N\": \[${t0_N}, ${tf_N}\]/" \
                    | sed "s/R\": \[3, 12\]/R\": \[$t, $tf\]/" \
                    | sed "s/ratio\"]       = True/ratio\"]       = ${ratio}/" \
                    > h-dibaryon_${ensemble}.py
                    python nn_fit.py
                else
                    echo "  already fit"
                fi
            done
        done
    done
done
