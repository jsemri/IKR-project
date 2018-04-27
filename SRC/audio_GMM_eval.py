#!/usr/bin/env python3
import matplotlib.pyplot as plt
from ikrlib import wav16khz2mfcc, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import scipy.linalg
import numpy as np
from numpy.random import randint
import os, sys
import pickle

def check_dir(folder):
    if not os.path.isdir(folder):
        raise RuntimeError('directory not found: ' + folder)

def main():
    check_dir('eval')
    test  = wav16khz2mfcc('eval')
    P_t = 0.5
    P_nt = 1 - P_t
    fname = 'GMM_model.pkl'
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    #choose one model
    with open(fname, 'rb') as f:
        Ws_t, MUs_t, COVs_t, Ws_nt, MUs_nt, COVs_nt = pickle.load(f)


    for tst in sorted(test.keys()):
        ll_t = logpdf_gmm(test[tst], Ws_t, MUs_t, COVs_t)
        ll_nt = logpdf_gmm(test[tst], Ws_nt, MUs_nt, COVs_nt)
        scr = (sum(ll_t) + np.log(P_t)) - (sum(ll_nt) + np.log(P_nt))
        tst = tst.split("/")[-1].split(".")[0]
        if scr >= 0:
            print(tst,scr,1);
        else:
            print(tst,scr,0);

main()
