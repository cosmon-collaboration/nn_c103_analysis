#!/bin/bash

for n in 1 2 3 4 5; do
    for t in $(seq 2 16); do
        echo "nstate= $n,  tmin= $t"
        if [[ ! -e result/N_n${n}_t_${t}-20.pickle ]]; then
            sed "s/\[8, 20\]/\[$t, 20\]/" n_parameters_base.py | sed "s/es\"\] = 1/es\"\] = $n/" > n_parameters.py
            python n_fit.py
        else
            echo "  already fit"
        fi
    done
done
