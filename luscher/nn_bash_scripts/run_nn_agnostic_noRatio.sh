#!/bin/bash

ratio="False"
t0=5
td=10
gevp="${t0}-${td}"
nn_iso='singlet'
tnorm=3
e=0
block=8

n_N=2
for n_NN in 2 3 4; do
    for t0_N in 4 5 7; do
        nucleon="n${n_N}_t_${t0_N}-20"
        for t in $(seq 2 9); do
            echo "estate= $e,  tmin= $t"
            result="result/NN_${nn_iso}_tnorm${tnorm}_t0-td_${gevp}_N_${nucleon}_NN_agnostic_n${n_NN}_e${e}_t_${t}-15_ratio_${ratio}_block${block}.pickle"
            echo $result
            if [[ ! -e $result ]]; then
                cp nn_parameters_base.py                                       nn_parameters.py
                sed -i '' "s/conspire/agnostic/"                               nn_parameters.py
                sed -i '' "s/triplet_S0/${nn_iso}_S0/"                         nn_parameters.py
                sed -i '' "s/t0\"\] = 5/t0\"\] = $t0/"                         nn_parameters.py 
                sed -i '' "s/td\"\] = 10/td\"\] = $td/"                        nn_parameters.py
                sed -i '' "s/ratio\"]       = True/ratio\"]       = ${ratio}/" nn_parameters.py
                sed -i '' "s/block\"\] = 2/block\"\] = $block/"                nn_parameters.py
                sed -i '' "s/nstates\"\]     = 3/nstates\"\]     = ${n_N}/"    nn_parameters.py 
                sed -i '' "s/r_n_inel\"\]    = 2/r_n_inel\"\]   = ${n_NN}/"    nn_parameters.py 
                sed -i '' "s/r_n_el\"\]      = 0/r_n_el\"\]     = $e/"         nn_parameters.py 
                sed -i '' "s/N\": \[3, 20\]/N\": \[${t0_N}, 20\]/"             nn_parameters.py
                sed -i '' "s/R\": \[3, 15\]/R\": \[$t, 15\]/"                  nn_parameters.py 
                python nn_fit.py
            else
                echo "  already fit"
            fi
        done
    done
done

n_N=3
for n_NN in 2 3 4 6; do
    for t0_N in 4 5 7; do
        nucleon="n${n_N}_t_${t0_N}-20"
        for t in $(seq 2 9); do
            echo "estate= $e,  tmin= $t"
            result="result/NN_${nn_iso}_tnorm${tnorm}_t0-td_${gevp}_N_${nucleon}_NN_agnostic_n${n_NN}_e${e}_t_${t}-15_ratio_${ratio}_block${block}.pickle"
            echo $result
            if [[ ! -e $result ]]; then
                cp nn_parameters_base.py                                       nn_parameters.py
                sed -i '' "s/conspire/agnostic/"                               nn_parameters.py
                sed -i '' "s/triplet_S0/${nn_iso}_S0/"                         nn_parameters.py
                sed -i '' "s/t0\"\] = 5/t0\"\] = $t0/"                         nn_parameters.py 
                sed -i '' "s/td\"\] = 10/td\"\] = $td/"                        nn_parameters.py
                sed -i '' "s/ratio\"]       = True/ratio\"]       = ${ratio}/" nn_parameters.py
                sed -i '' "s/block\"\] = 2/block\"\] = $block/"                nn_parameters.py
                sed -i '' "s/nstates\"\]     = 3/nstates\"\]     = ${n_N}/"    nn_parameters.py 
                sed -i '' "s/r_n_inel\"\]    = 2/r_n_inel\"\]   = ${n_NN}/"    nn_parameters.py 
                sed -i '' "s/r_n_el\"\]      = 0/r_n_el\"\]     = $e/"         nn_parameters.py 
                sed -i '' "s/N\": \[3, 20\]/N\": \[${t0_N}, 20\]/"             nn_parameters.py
                sed -i '' "s/R\": \[3, 15\]/R\": \[$t, 15\]/"                  nn_parameters.py 
                python nn_fit.py
            else
                echo "  already fit"
            fi
        done
    done
done
