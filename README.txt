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

Put all train directories into data directory 
and all evaluation data into eval directory.

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

./img_learn.py [SGD]

    If SGD argument is set, model will be SGDClassifier.
    If there is no argument, model will be RandomForestClassifier.

Script will train given classifier on data and trained model save to model_experiment file.

2.2.2 Audio

./audio_EM.py

Script will train GMM classifier on data and trained model save to GMM_model.pkl

2.3 Evaluate models

2.3.1 Images

./img_prob.py [file_name]
    
    If file_name is set, reads model from given file
    If there is no argument, reads model from file model_experiment

Script will classify images from eval directory and prints results on stdout

2.3.2 Audio

./audio_GMM_eval.py

Script will classify audio from eval directory and prints results on stdout 
