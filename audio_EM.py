#!/usr/bin/env python3

import matplotlib.pyplot as plt
from ikrlib import wav16khz2mfcc, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm, gellipse
import scipy.linalg
import numpy as np
from numpy.random import randint
import os
import pickle

negative_train_path = 'data/non_target_train'
positive_train_path = 'data/target_train'
negative_test_path = 'data/non_target_dev'
positive_test_path = 'data/target_dev'

def check_dir(folder):
    if not os.path.isdir(folder):
        raise RuntimeError('directory not found: ' + folder)

def main():
    check_dir(os.path.dirname(negative_test_path))
    check_dir(os.path.dirname(negative_train_path))
    check_dir(os.path.dirname(positive_test_path))
    check_dir(os.path.dirname(positive_train_path))

    train_m = list(wav16khz2mfcc(positive_train_path).values())
    train_f = list(wav16khz2mfcc(negative_train_path).values())
    test_m  = list(wav16khz2mfcc(positive_test_path).values())
    test_f  = list(wav16khz2mfcc(negative_test_path).values())

    train_m = np.vstack(train_m)
    train_f = np.vstack(train_f)
    dim = train_m.shape[1]

    cov_tot = np.cov(np.vstack([train_m, train_f]).T, bias=True)
    d, e = scipy.linalg.eigh(cov_tot, eigvals=(dim-2, dim-1))

    train_m_pca = train_m.dot(e)
    train_f_pca = train_f.dot(e)
    # Classes are not well separated in 2D PCA subspace

    n_m = len(train_m)
    n_f = len(train_f)
    cov_wc = (n_m*np.cov(train_m.T, bias=True) + n_f*np.cov(train_f.T, bias=True)) / (n_m + n_f)
    cov_ac = cov_tot - cov_wc
    d, e = scipy.linalg.eigh(cov_ac, cov_wc, eigvals=(dim-1, dim-1))

    # Lets define uniform a-priori probabilities of classes:
    P_m = 0.5
    P_f = 1 - P_m


    ll_m = logpdf_gauss(test_m[0], np.mean(train_m, axis=0), np.var(train_m, axis=0))
    ll_f = logpdf_gauss(test_m[0], np.mean(train_f, axis=0), np.var(train_f, axis=0))

    posterior_m = np.exp(ll_m)*P_m /(np.exp(ll_m)*P_m + np.exp(ll_f)*P_f)


    ll_m = logpdf_gauss(test_m[0], *train_gauss(train_m))
    ll_f = logpdf_gauss(test_m[0], *train_gauss(train_f))
    # '*' before 'train_gauss' pases both return values (mean and cov) as parameters of 'logpdf_gauss'
    posterior_m = np.exp(ll_m)*P_m /(np.exp(ll_m)*P_m + np.exp(ll_f)*P_f);
    plt.figure(); plt.plot(posterior_m, 'b'); plt.plot(1-posterior_m, 'r');
    plt.figure(); plt.plot(ll_m, 'b');        plt.plot(ll_f, 'r');


    # Again gaussian models with full covariance matrices. Now testing a female utterance

    ll_m = logpdf_gauss(test_f[1], *train_gauss(train_m))
    ll_f = logpdf_gauss(test_f[1], *train_gauss(train_f))
    # '*' before 'train_gauss' pases both return values (mean and cov) as parameters of 'logpdf_gauss'
    posterior_m = np.exp(ll_m)*P_m /(np.exp(ll_m)*P_m + np.exp(ll_f)*P_f);
    plt.figure(); plt.plot(posterior_m, 'b'); plt.plot(1-posterior_m, 'r');
    plt.figure(); plt.plot(ll_m, 'b');        plt.plot(ll_f, 'r');

    score=[]
    mean_m, cov_m = train_gauss(train_m)
    mean_f, cov_f = train_gauss(train_f)
    for tst in test_m:
        ll_m = logpdf_gauss(tst, mean_m, cov_m)
        ll_f = logpdf_gauss(tst, mean_f, cov_f)
        score.append((sum(ll_m) + np.log(P_m)) - (sum(ll_f) + np.log(P_f)))

    # Run recognition with 1-dimensional LDA projected data
    score=[]
    mean_m, cov_m = train_gauss(train_m.dot(e))
    mean_f, cov_f = train_gauss(train_f.dot(e))
    for tst in test_m:
        ll_m = logpdf_gauss(tst.dot(e), mean_m, np.atleast_2d(cov_m))
        ll_f = logpdf_gauss(tst.dot(e), mean_f, np.atleast_2d(cov_f))
        score.append((sum(ll_m) + np.log(P_m)) - (sum(ll_f) + np.log(P_f)))

    M_m = 12

    MUs_m  = train_m[randint(1, len(train_m), M_m)]
    COVs_m = [np.var(train_m, axis=0)] * M_m
    Ws_m   = np.ones(M_m) / M_m;

    M_f = 7
    MUs_f  = train_f[randint(1, len(train_f), M_f)]
    COVs_f = [np.var(train_f, axis=0)] * M_f
    Ws_f   = np.ones(M_f) / M_f;

    for jj in range(100):
        [Ws_m, MUs_m, COVs_m, TTL_m] = train_gmm(train_m, Ws_m, MUs_m, COVs_m);
        [Ws_f, MUs_f, COVs_f, TTL_f] = train_gmm(train_f, Ws_f, MUs_f, COVs_f);
        plt.plot(train_f_pca[:,1], train_f_pca[:,0], 'r.', ms=1)
        plt.plot(train_m_pca[:,1], train_m_pca[:,0], 'b.', ms=1)
        for w, m, c in zip(Ws_f, MUs_f, COVs_f): gellipse(m, c, 100, 'r', lw=round(w * 10))
        for w, m, c in zip(Ws_m, MUs_m, COVs_m): gellipse(m, c, 100, 'b', lw=round(w * 10))
        plt.show()

    score=[]
    testok = 0
    testnok = 0
    for tst in test_m:
        ll_m = logpdf_gmm(tst, Ws_m, MUs_m, COVs_m)
        ll_f = logpdf_gmm(tst, Ws_f, MUs_f, COVs_f)
        scr = (sum(ll_m) + np.log(P_m)) - (sum(ll_f) + np.log(P_f))
        score.append(scr)
        if scr >= 0:
            testok += 1
        else:
            testnok += 1
    print("target is " + str(testok/(testok+testnok)))

    score=[]
    testok = 0
    testnok = 0
    for tst in test_f:
        ll_m = logpdf_gmm(tst, Ws_m, MUs_m, COVs_m)
        ll_f = logpdf_gmm(tst, Ws_f, MUs_f, COVs_f)
        scr = (sum(ll_m) + np.log(P_m)) - (sum(ll_f) + np.log(P_f))
        score.append(scr)
        if scr < 0:
            testok += 1
        else:
            testnok += 1
    print("non target is " + str(testok/(testok+testnok)))
    print('Saved as "GMM_model.pkl"')
    with open('GMM_model.pkl', 'wb') as f:
        pickle.dump([Ws_m, MUs_m, COVs_m, Ws_f, MUs_f, COVs_f], f)

main()
