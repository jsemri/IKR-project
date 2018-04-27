IKR-project
IKR team school project

1. Install

You can install all requirements with `pip` or with python official supported `pipenv`

1.1 PIPENV

$ pipenv install


1.2 PIP3

$ pip3 install -r /path/to/requirements.txt

2. Usage

2.1 Expected directory structure

Put all train directories into the data directory
and all evaluation data into the eval directory.

eval and data directory should be in the same directory
as python scripts (SRC).

-- SRC
   |-- eval
   |-- data
       |-- target_dev
       |-- non_target_dev
       |-- target_train
       |-- non_target_train

2.2 Train models

2.2.1 Images

./SRC/img_learn.py [SGD]

    Script trains Random Forest ensemble model on the dataset in data directory.
    If any argument is present, SGD classifier is set as a model.

Script trains a given classifier on data and saves the model to model_experiment file.

2.2.2 Audio

./SRC/audio_EM.py

Script trains a GMM classifier on data and saves it to GMM_model.pkl file.

2.3 Evaluate models

2.3.1 Images

./SRC/img_prob.py [file_name]

    If file_name is set, read a model from a given file
    If there is no argument, read a model from the file named model_experiment

Script classifies images from eval directory and prints results on stdout

2.3.2 Audio

./SRC/audio_GMM_eval.py

Script classifies audio from eval directory and prints results on stdout
