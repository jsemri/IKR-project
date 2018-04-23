#!/usr/bin/env python3

from glob import glob
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

from image_classifier import read_img

def score(model, path):
    X = read_img(path)
    X = X.reshape(X.shape[0],-1)
    preds = model.predict_proba(X)
    print(preds)

def main():
    model = joblib.load('img_cls_model2')
    path1 = 'data/non_target_dev/*.png'
    path2 = 'data/target_dev/*.png'
    score(model, path1)

main()