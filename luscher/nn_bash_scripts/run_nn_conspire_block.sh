#!/bin/bash

ratio="False"
nn_iso='singlet'
e=0

#for gevp_t in "5-10" "5-12" "6-10" "6-12"; do
gevp_t="5-10"
for n_N in 3 4; do 
    t0=${gevp_t%-*}
    td=${gevp_t#*-}
    gevp="${t0}-${td}"
    echo "$t0, $td, $gevp_t, $gevp"
    for block in 1 2 5 10; do
        for t0_N in 2 3 4 5; do
            nucleon="n${n_N}_t_${t0_N}-20"
            for t in $(seq 2 11); do
                if [[ $block -eq 1 ]]; then
                    result="result/NN_${nn_iso}_t0-td_${gevp}_N_${nucleon}_NN_conspire_e${e}_t_${t}-15_ratio_${ratio}_bsPrior-gs.pickle"
                else
                    result="result/NN_${nn_iso}_t0-td_${gevp}_N_${nucleon}_NN_conspire_e${e}_t_${t}-15_ratio_${ratio}_block${block}_bsPrior-gs.pickle"
                fi
                echo ""
                echo $result
                if [[ ! -e $result ]]; then
                    sed   "s/triplet_S0/${nn_iso}_S0/" nn_parameters_base.py \
                    | sed "s/t0\"\] = 5/t0\"\] = ${t0}/" \
                    | sed "s/td\"\] = 10/td\"\] = ${td}/" \
                    | sed "s/block\"\] = 2/block\"\] = $block/" \
                    | sed "s/nstates\"]     = 3/nstates\"]     = ${n_N}/" \
                    | sed "s/N\": \[3, 20\]/N\": \[${t0_N}, 20\]/" \
                    | sed "s/R\": \[3, 15\]/R\": \[$t, 15\]/" \
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
