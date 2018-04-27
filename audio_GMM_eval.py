import matplotlib.pyplot as plt
from ikrlib import wav16khz2mfcc, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import scipy.linalg
import numpy as np
from numpy.random import randint
from audio_processing import *

test  = wav16khz2mfcc('data/eval')

P_t = 0.5
P_nt = 1 - P_t


import pickle
#choose one model
with open('audio_GMM_model1.pkl', 'rb') as f:
#with open('audio_GMM_model2.pkl', 'rb') as f:
    Ws_t, MUs_t, COVs_t, Ws_nt, MUs_nt, COVs_nt = pickle.load(f)


#and filename
with open('audio_GMM_result.txt', 'w') as rfile:
#with open('audio_GMM_result2.txt', 'w') as rfile:

    for tst in test.keys():
        ll_t = logpdf_gmm(test[tst], Ws_t, MUs_t, COVs_t)
        ll_nt = logpdf_gmm(test[tst], Ws_nt, MUs_nt, COVs_nt)
        scr = (sum(ll_t) + np.log(P_t)) - (sum(ll_nt) + np.log(P_nt))
        tst = tst.split("/")[-1].split(".")[0]
        if scr >= 0:
            rfile.write(tst + " " + str(scr) + " 1\n");
        else:
            rfile.write(tst + " " + str(scr) + " 0\n");
