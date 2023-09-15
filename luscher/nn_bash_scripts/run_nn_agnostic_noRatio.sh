#!/bin/bash

ratio="False"

t0=5
td=10
gevp="${t0}-${td}"
nn_iso='singlet'
e=0

for n_N in 2 3; do
    for n_NN in 2 3; do
        for t0_N in 2 5 7; do
            nucleon="n${n_N}_t_${t0_N}-20"
            for t in $(seq 2 11); do
                echo "estate= $e,  tmin= $t"
                result="result/NN_${nn_iso}_t0-td_${gevp}_N_${nucleon}_NN_agnostic_n${n_NN}_e${e}_t_${t}-15_ratio_${ratio}.pickle"
                echo $result
                if [[ ! -e $result ]]; then
                    cp nn_parameters_base.py                                      nn_parameters.py
                    sed -i '' "s/triplet_S0/${nn_iso}_S0/g"                       nn_parameters.py
                    sed -i '' "s/t0\"\] = 5/t0\"\] = $t0/g"                       nn_parameters.py 
                    sed -i '' "s/td\"\] = 10/td\"\] = $td/g"                      nn_parameters.py
                    sed -i '' "s/ratio\"]      = True/ratio\"]      = ${ratio}/g" nn_parameters.py
                    sed -i '' "s/nstates\"\]    = 2/nstates\"\]    = ${n_N}/g"    nn_parameters.py 
                    sed -i '' "s/r_n_inel\"\]   = 2/r_n_inel\"\]   = ${n_NN}/g"   nn_parameters.py 
                    sed -i '' "s/r_n_el\"\]     = 0/r_n_el\"\]     = $e/g"        nn_parameters.py 
                    sed -i '' "s/N\": \[5, 20\]/N\": \[${t0_N}, 20\]/"            nn_parameters.py
                    sed -i '' "s/R\": \[6, 15\]/R\": \[$t, 15\]/g"                nn_parameters.py 
                    python nn_fit.py
                else
                    echo "  already fit"
                fi
            done
        done
    done
done
