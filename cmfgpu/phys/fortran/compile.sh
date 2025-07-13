#!/bin/bash

cd "$(dirname "$0")"

must_file="parkind1.F90"

for f90 in *.F90; do
    if [ "$f90" = "$must_file" ]; then
        continue
    fi
    module_name="${f90%.F90}"
    cmd="python -m numpy.f2py -c $must_file $f90 -m $module_name"
    $cmd
done