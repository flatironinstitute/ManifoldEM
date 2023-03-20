#!/bin/sh

# Copyright (c) Columbia University Evan Seitz 2019

workDir=$(pwd)
scriptDir="$( cd "$(dirname "$0")" ; pwd -P )"

cd ..
cd ..
cd bin

for filename in *50.star
do
    echo "input: $filename"
    echo "output: ${workDir}/${filename%.*}.mrc" 
    relion_reconstruct --i "${filename}" --o "${workDir}/${filename%.*}.mrc" &
done
