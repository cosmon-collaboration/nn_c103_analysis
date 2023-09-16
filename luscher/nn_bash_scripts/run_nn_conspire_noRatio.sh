#!/bin/bash

#n_N=2
#t0_N=5
#nucleon="n${n_N}_t_${t0_N}-20"

ratio="False"

t0=3
td=8
gevp="${t0}-${td}"
nn_iso='singlet'

for n_N in 2 3 4; do
    for t0_N in 2 5 7; do
        e=0
        nucleon="n${n_N}_t_${t0_N}-20"
        for t in $(seq 2 11); do
            echo "estate= $e,  tmin= $t"
            echo result/NN_${nn_iso}_t0-td_${gevp}_N_${nucleon}_NN_conspire_e${e}_t_${t}-15_ratio_${ratio}.pickle
            if [[ ! -e result/NN_${nn_iso}_t0-td_${gevp}_N_${nucleon}_NN_conspire_e${e}_t_${t}-15_ratio_${ratio}.pickle ]]; then
                sed "s/R\": \[6, 15\]/R\": \[$t, 15\]/" nn_parameters_base.py \
                | sed "s/t0\"\] = 5/t0\"\] = ${t0}/" \
                | sed "s/td\"\] = 10/td\"\] = ${td}/" \
                | sed "s/nstates\"]    = 2/nstates\"]    = ${n_N}/" \
                | sed "s/N\": \[5, 20\]/N\": \[${t0_N}, 20\]/" \
                | sed "s/r_n_el\"\]     = 0/r_n_el\"\]     = $e/" \
                | sed "s/triplet_S0/${nn_iso}_S0/" \
                | sed "s/ratio\"]      = True/ratio\"]      = ${ratio}/" \
                | sed "s/agnostic/conspire/" > nn_parameters.py
                python nn_fit.py
            else
                echo "  already fit"
            fi
        done
    done
done
