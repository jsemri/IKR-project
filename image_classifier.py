#!/usr/bin/env python3

from glob import glob
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier # not used, higher error

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from scipy import misc

def read_img(path):
    return np.array([np.array(misc.imread(f)) for f in glob(path)])

def get_data():
    # paths
    negative_train_path = 'data/non_target_train/*.png'
    target_train_path = 'data/target_train/*.png'
    negative_test_path = 'data/non_target_dev/*.png'
    target_test_path = 'data/target_dev/*.png'

    # test data
    x_test = read_img(target_test_path)
    y_test = np.array([1]*len(x_test))

    t = read_img(negative_test_path)
    y_test = np.concatenate((y_test,np.array([0]*len(t))),axis=0)
    x_test = np.concatenate((x_test, t),axis=0)

    # train data
    x_train = read_img(target_train_path)
    y_train = np.array([1]*len(x_train))

    t = read_img(negative_train_path)
    y_train = np.concatenate((y_train,np.array([0]*len(t))),axis=0)
    x_train = np.concatenate((x_train, t),axis=0)

    #print('Train shape: ', x_train.shape)
    #print('Test shape: ', x_test.shape)

    x_train, y_train = shuffle(x_train, y_train, random_state=21)
    x_test, y_test = shuffle(x_test, y_test, random_state=21)

    return x_train, y_train, x_test, y_test

def cross_validation(X,y):
    pass

def display_errors(x_test,y_test,preds):
    for idx,im in enumerate(x_test):
        if y_test[idx] != preds[idx]:
            im = im.reshape(80,80,3)
            plt.imshow(im)
            print(preds[idx])
            plt.show()

def train_model(model,x_train,y_train,normal=False,epochs=5):
    datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.1,
        height_shift_range=0.1,shear_range=0.2,
        horizontal_flip=True)

    if not normal:
        for e in range(epochs):
            i = 0
            for x_batch, y_batch in datagen.flow(x_train,y_train,batch_size=128):
                sh = x_batch.shape
                #print(y_batch)
                x_batch.shape = (sh[0],sh[1]*sh[2]*sh[3])
                model.fit(x_batch,y_batch)
                i += 1
                if i > 10: break
    else:
        sh = x_train.shape
        x_train.shape = (sh[0],sh[1]*sh[2]*sh[3])
        model.fit(x_train,y_train)

def validate_model(model,x_test,y_test,normal=False,epochs=5):
    datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.1,
        height_shift_range=0.1,shear_range=0.2,
        horizontal_flip=True)

    if normal:
        sh = x_test.shape
        x_test.shape = (sh[0],sh[1]*sh[2]*sh[3])
        preds = model.predict(x_test)
        print('accuracy: ', accuracy_score(y_test, preds))
        #preds = model.predict_proba(x_test)
        #print(preds)
    else:
        pr = []
        target_in = []
        for e in range(epochs):
            i = 0
            for x_batch, y_batch in datagen.flow(x_test,y_test,batch_size=128):
                sh = x_batch.shape
                x_batch.shape = (sh[0],sh[1]*sh[2]*sh[3])
                preds = model.predict(x_batch)
                pr.append(accuracy_score(y_batch, preds))
                target_in.append(np.mean(y_batch))
                i += 1
                if i > 10: break

        print('average: ',np.mean(pr))
        print('variance: ', np.std(pr))
        print('negative odds: ', 1 - np.mean(target_in))

def main():
    x_train, y_train, x_test, y_test = get_data()

    model = RandomForestClassifier()
    #model = SGDClassifier(loss='hinge',penalty='l2')
    train_model(model,x_train,y_train,0)
    validate_model(model,x_test,y_test,0)


if __name__ == "__main__":
    main()
