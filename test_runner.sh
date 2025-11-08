#!/bin/bash

for n in {1..20}; do
    echo "Running with n = $n"
    mpirun --oversubscribe -n "$n" python tests.py
    echo "----------------------------------------"
done