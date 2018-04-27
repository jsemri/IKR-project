#!/usr/bin/env python3

from glob import glob
import re
import sys, os
import numpy as np
from scipy import misc
import pickle

def score(model, path):
    fnames = sorted(glob(path))
    assert len(fnames) > 0
    for fname in fnames:
        img = np.array(misc.imread(fname)).flatten()
        preds = model.predict_proba([img])
        preds = preds[0]
        print(re.sub('\..*','', os.path.basename(fname)), preds[1],
            int(preds[0] < preds[1]))
def main():
    if not os.path.isdir('eval'):
        raise RuntimeError('directory not found: eval')

    fname = 'model_experiment'
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    with open(fname,'rb') as f:
        model = pickle.load(f)
    # print: filename soft-decision hard-decision
    score(model, 'eval/*.png')

main()
