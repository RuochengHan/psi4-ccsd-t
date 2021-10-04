#! /bin/bash

nmo= # number of MO (not SO)
nocc= # number of occupied MO (not SO)

python read_F_ERI.py $FOCK $ERI2
mv ERI.npy ERI2.npy
python read_F_ERI.py $FOCK $ERI1
python CCSD_pytorch_read_F_ERI2.py $nmo $nocc
