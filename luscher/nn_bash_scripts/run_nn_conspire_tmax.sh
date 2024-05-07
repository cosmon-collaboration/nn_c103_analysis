#!/bin/bash

ratio="False"
nn_iso='singlet'
tnorm=3
e=0
block=8

#for gevp_t in "5-10" "5-12" "6-10" "6-12"; do
for gevp_t in "5-10"; do
    t0=${gevp_t%-*}
    td=${gevp_t#*-}
    gevp="${t0}-${td}"
    for n_N in 3; do
        for t in 3 4 5 6 7; do
            for tf_NN in 10 11 12 13 14 15; do
                for t0_N in 3 4 5; do
                    nucleon="n${n_N}_t_${t0_N}-20"
                    nn="conspire_e${e}_t_${t}-${tf_NN}_ratio_${ratio}"
                    result="result/NN_${nn_iso}_tnorm${tnorm}_t0-td_${gevp}_N_${nucleon}_NN_${nn}_block${block}.pickle"
                    echo ""
                    echo $result
                    if [[ ! -e $result ]]; then
                        sed   "s/triplet_S0/${nn_iso}_S0/" nn_parameters_base.py \
                        | sed "s/t0\"\] = 5/t0\"\] = ${t0}/" \
                        | sed "s/td\"\] = 10/td\"\] = ${td}/" \
                        | sed "s/t_norm\'\] = 3/t_norm\'\] = ${tnorm}/" \
                        | sed "s/nstates\"]     = 3/nstates\"]     = ${n_N}/" \
                        | sed "s/N\": \[3, 20\]/N\": \[${t0_N}, 20\]/" \
                        | sed "s/R\": \[3, 15\]/R\": \[$t, $tf_NN\]/" \
                        | sed "s/ratio\"]       = True/ratio\"]       = ${ratio}/" \
                        > nn_parameters.py
                        python nn_fit.py
                    else
                        echo "  already fit"
                    fi
                done
            done
        done
    done
done
