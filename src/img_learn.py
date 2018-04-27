#!/usr/bin/env python3

from glob import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from scipy import misc

from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

negative_train_path = 'data/non_target_train/*.png'
positive_train_path = 'data/target_train/*.png'
negative_test_path = 'data/non_target_dev/*.png'
positive_test_path = 'data/target_dev/*.png'

def read_img(path):
    return np.array([np.array(misc.imread(f)) for f in glob(path)])

def get_data():
    # test data
    x_test = read_img(positive_test_path)
    y_test = np.array([1]*len(x_test))

    t = read_img(negative_test_path)
    y_test = np.concatenate((y_test,np.array([0]*len(t))),axis=0)
    x_test = np.concatenate((x_test, t),axis=0)

    # train data
    x_train = read_img(positive_train_path)
    y_train = np.array([1]*len(x_train))

    t = read_img(negative_train_path)
    y_train = np.concatenate((y_train,np.array([0]*len(t))),axis=0)
    x_train = np.concatenate((x_train, t),axis=0)

    x_train, y_train = shuffle(x_train, y_train, random_state=21)
    x_test, y_test = shuffle(x_test, y_test, random_state=21)

    return x_train, y_train, x_test, y_test

def cross_val(model,display=False):
    X1, y1, X2, y2 = get_data()
    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    dg = ImageDataGenerator(rotation_range=20,width_shift_range=0.1,
        height_shift_range=0.1,shear_range=0.2, horizontal_flip=True)
    train_x, train_y = [], []
    i = 0
    for x_batch, y_batch in dg.flow(X,y,batch_size=1000):
        train_x.append(x_batch)
        train_y.append(y_batch)
        i += 1
        if i > 10: break

    # concatenate arrays
    y = np.hstack(train_y)
    X = np.vstack(train_x)
    X = X.reshape(X.shape[0],-1)

    res = cross_val_score(model, X, y, cv=KFold(n_splits=5,random_state=7))
    print('Folds: ', res)
    print('mean: ', res.mean(), 'std: ', res.std())
    # display the result of cross-validation
    res *= 100
    if display:
        base = min((85,min(res)-1))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(['fold ' + str(i+1) for i in range(len(res))], res-base, bottom=85)
        ax.axhline(y=res.mean(), color='red', linestyle='--')
        ax.set_title('cross-validation summary')
        ax.set_ylabel('accuracy')
        ax.set_yticks(np.arange(base,100,1))
        plt.show()

    # split to avoid over-fitting or perhaps train on the whole dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
        random_state=78)
    model.fit(x_train,y_train)
    print(accuracy_score(model.predict(x_test), y_test))

    # save the model
    with open('model_experiment','wb') as f:
        pickle.dump(model, f)

def check_dir(folder):
    if not os.path.isdir(folder):
        raise RuntimeError('directory not found: ' + folder)

def main():
    check_dir(os.path.dirname(negative_test_path))
    check_dir(os.path.dirname(negative_train_path))
    check_dir(os.path.dirname(positive_test_path))
    check_dir(os.path.dirname(positive_train_path))

    model = RandomForestClassifier()
    # any argument changes the model type to SGD
    if len(sys.argv) > 1:
        print('Using SGD classifier.')
        #loss = 'log'
        loss = 'modified_huber'
        model = SGDClassifier(loss=loss, penalty='l2')

    cross_val(model,display=True)

if __name__ == "__main__":
    main()
