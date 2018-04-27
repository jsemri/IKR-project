#!/usr/bin/env python3

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.metrics import accuracy_score
import pickle
import sys

from image_classifier import read_img

np.set_printoptions(threshold=np.nan)

def score(model, path):
    fnames = sorted(glob(path))
    nm = set()
    c = 0
    for f in fnames:
        c += 1
        img = np.array(misc.imread(f)).flatten()
        preds = model.predict_proba([img])
        if preds[0][0] < preds[0][1]:
            nm.add(int(f[-7:-4]))

    with open('eval_targets','r') as f:
        ev = set(int(i) for i in f.read().split())
    p = len(ev)
    n = c - p
    tp = len(ev & nm)
    tn = n - len(nm - ev)
    fp = len(nm - ev)
    fn = len(ev - nm)
    print('TP:', tp)
    print('FP:', fp)
    print('FN:', fn)
    print('TN:', tn)
    print('accuracy:', (tp+tn)/(c))
    print('precision:', (tp)/(tp+fp))

def main():
    fname = 'model_experiment'
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    with open(fname,'rb') as f:
        model = pickle.load(f)
    path1 = 'data/non_target_dev/*.png'
    path2 = 'data/target_dev/*.png'
    path3 = 'eval/*.png'
    score(model, path3)

main()
