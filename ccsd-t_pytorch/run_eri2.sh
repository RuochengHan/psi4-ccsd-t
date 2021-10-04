#! /bin/bash

nmo=
nocc=

python read_F_ERI.py fockm terimol
mv ERI.npy ERI2.npy
python read_F_ERI.py fockm ferimol
python CCSD_pytorch_read_F_ERI2.py $nmo $nocc
