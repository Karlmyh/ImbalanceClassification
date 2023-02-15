import os
from time import time
import numpy as np
import pandas as pd
from itertools import product

from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from OKNN import OKNN

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score

# config
dataset_dir = "./data/"
dataset_name_seq = [
    #"OccupancyDetection",
    #"Adult",
    #"APSFailure",
    "BuzzInSocialMedia-Twitter_Relative_labeling_sigma1000",
    "DefaultOfCreditCardClients",
    #"BitcoinHeistRansomwareAddress-binarize",
]

classifier_method = "OKNN"

n_neighbors_seq = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]
n_neighbors_density_seq = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]

parameters = {"n_neighbors": n_neighbors_seq, "n_neighbors_density": n_neighbors_density_seq}

n_jobs = 32
n_splits = 10
n_repeats = 2
random_state = 101

save_dir = "./result/realdataresult/"
os.makedirs(save_dir, exist_ok=True)
save_name = "{}.csv".format(classifier_method)
save_path = os.path.join(save_dir, save_name)
error_name = "{}_error.csv".format(classifier_method)
error_path = os.path.join(save_dir, error_name)

for dataset_name in dataset_name_seq:
    RepeatRandomState = np.random.RandomState(random_state)
    dataset_path = os.path.join(dataset_dir, "{}.csv".format(dataset_name))
    data = pd.read_csv(dataset_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X = np.array(X)
    y = np.array(y)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits,
                                   n_repeats=n_repeats,
                                   random_state=random_state)
    for idx, (train_index, test_index) in enumerate(rskf.split(X, y)):
        repeat_state = RepeatRandomState.randint(0, np.iinfo(np.uint32).max)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # scaling dataset
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # running
        
        try:
            test_time_max = 0
            test_am_max = 0
            n_neighbors_max = 0
            n_neighbors_density_max = 0
            for n_neighbors, n_neighbors_density in product(n_neighbors_seq, n_neighbors_density_seq):

                model_OKNN = OKNN(n_neighbors = n_neighbors, n_neighbors_density = n_neighbors_density)
                model_OKNN.fit(X_train, y_train)

                time_start = time()
                y_test_hat = model_OKNN.predict(X_test)
                time_end = time()

                test_time = time_end - time_start

                test_am = recall_score(y_true=y_test,
                                           y_pred=y_test_hat,
                                           average="macro") 
                if test_am > test_am_max:
                    test_am_max = test_am
                    test_time_max = test_time
                    n_neighbors_density_max = n_neighbors_density
                    n_neighbors_max = n_neighbors
            # save results
            with open(save_path, 'a') as f:
                f.writelines(
                    "{},{},{},{},{},{}\n"
                    .format(dataset_name, idx, test_time_max, test_am_max,n_neighbors_max,n_neighbors_density_max))
        except Exception as e:
            with open(error_path, "a") as f:
                f.writelines("{},{},{}\n".format(
                    dataset_name, idx, e))
