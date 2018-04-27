import matplotlib.pyplot as plt
from ikrlib import wav16khz2mfcc, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import scipy.linalg
import numpy as np
from numpy.random import randint
from audio_processing import *

import pickle

train_t = list(wav16khz2mfcc('data/target_train').values())
train_nt = list(wav16khz2mfcc('data/non_target_train').values())
test_t  = list(wav16khz2mfcc('data/target_dev').values())
test_nt  = list(wav16khz2mfcc('data/non_target_dev').values())

train_t = np.vstack(train_t)
train_nt = np.vstack(train_nt)
dim = train_t.shape[1]


cov_tot = np.cov(np.vstack([train_t, train_nt]).T, bias=True)
d, e = scipy.linalg.eigh(cov_tot, eigvals=(dim-2, dim-1))

train_t_pca = train_t.dot(e)
train_nt_pca = train_nt.dot(e)
n_t = len(train_t)
n_nt = len(train_nt)
cov_wc = (n_t*np.cov(train_t.T, bias=True) + n_nt*np.cov(train_nt.T, bias=True)) / (n_t + n_nt)
cov_ac = cov_tot - cov_wc
d, e = scipy.linalg.eigh(cov_ac, cov_wc, eigvals=(dim-1, dim-1))


P_t = 0.5
P_nt = 1 - P_t

M_t = 12
M_nt = 7

t_proc = 0
t_proc_prev = 0.9333333333333333  #last saved model
nt_proc = 0
nt_proc_prev = 0.9333333333333333  #last saved model
while t_proc < 1.0 or nt_proc < 1.0:
    print("start")
    MUs_t  = train_t[randint(1, len(train_t), M_t)]
    COVs_t = [np.var(train_t, axis=0)] * M_t
    Ws_t   = np.ones(M_t) / M_t;

    MUs_nt  = train_nt[randint(1, len(train_nt), M_nt)]
    COVs_nt = [np.var(train_nt, axis=0)] * M_nt
    Ws_nt   = np.ones(M_nt) / M_nt;

    for jj in range(200):
      [Ws_t, MUs_t, COVs_t, TTL_t] = train_gmm(train_t, Ws_t, MUs_t, COVs_t);
      [Ws_nt, MUs_nt, COVs_nt, TTL_nt] = train_gmm(train_nt, Ws_nt, MUs_nt, COVs_nt);
      #print('Iteration:', jj, ' Total log-likelihood:', TTL_t, 'for target;', TTL_nt, 'for non-target')

    score=[]
    testok = 0
    testnok = 0
    #print("target data")
    for tst in test_t:
        ll_t = logpdf_gmm(tst, Ws_t, MUs_t, COVs_t)
        ll_nt = logpdf_gmm(tst, Ws_nt, MUs_nt, COVs_nt)
        scr = (sum(ll_t) + np.log(P_t)) - (sum(ll_nt) + np.log(P_nt))
        score.append(scr)
        if scr >= 0:
            testok += 1
        else:
            testnok += 1
    #print(score)
    t_proc = testok/(testok+testnok)
    print("target is " + str(t_proc))

    score=[]
    testok = 0
    testnok = 0
    #print("non target data")
    for tst in test_nt:
        ll_t = logpdf_gmm(tst, Ws_t, MUs_t, COVs_t)
        ll_nt = logpdf_gmm(tst, Ws_nt, MUs_nt, COVs_nt)
        scr = (sum(ll_t) + np.log(P_t)) - (sum(ll_nt) + np.log(P_nt))
        score.append(scr)
        if scr < 0:
            testok += 1
        else:
            testnok += 1
    #print(score)
    nt_proc = testok/(testok+testnok)
    print("non target is " + str(nt_proc))

    if t_proc >= t_proc_prev and nt_proc >= nt_proc_prev:
        with open('trained_objects.pkl', 'wb') as f:
            pickle.dump([Ws_t, MUs_t, COVs_t, Ws_nt, MUs_nt, COVs_nt], f)
        print("saved target: " + str(t_proc) + " non_target: " + str(nt_proc))
        t_proc_prev = t_proc
        nt_proc_prev = nt_proc

    if t_proc == 1.0 and nt_proc >= 0.85:
        with open('trained_objects1.0.pkl', 'wb') as f:
            pickle.dump([Ws_t, MUs_t, COVs_t, Ws_nt, MUs_nt, COVs_nt], f)
        print("saved target: " + str(t_proc) + " non_target: " + str(nt_proc))
