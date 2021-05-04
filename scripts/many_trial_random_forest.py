import csv, datetime, joblib, time, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from dask.distributed import Client, progress


client = Client(processes=False, threads_per_worker=6, n_workers=1, memory_limit='10GB')

N_PER_TREE_CHOICE = 4
MIN_TREES = 1
MAX_TREES = 6

ml_df = pd.read_pickle("../data/Cleaned_Chicago_Sales.pkl")
label_cols = ["Sale Price"]

labels = np.array(ml_df[label_cols])
features = ml_df.drop(label_cols, axis=1)
feature_list = list(features.columns)
features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(np.nan_to_num(features), labels, test_size = 0.25, random_state = 42)
train_labels, test_labels = train_labels.ravel(), test_labels.ravel()
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

if not os.path.exists("./output"):
    os.mkdir("./output")

output_file = "./output/RFR-{}trials-[{}-{}trees]-{}.csv".format(N_PER_TREE_CHOICE,int(2**MIN_TREES), int(2**MAX_TREES), datetime.datetime.now())

with open(output_file, "w") as out:
    csv_out = csv.writer(out)
    csv_out.writerow([ "NTree", "Trial", "MAE", "Accuracy" ])
    for num_trees in range(MIN_TREES,MAX_TREES):
        t0 = time.time()
        avg_tree_err = 0
        for trial in range(N_PER_TREE_CHOICE):
            print("NTree: {:4d}, Trial {:3d}".format(int(2**num_trees), trial))
            rf = RandomForestRegressor(n_estimators = int(2**num_trees))
            with joblib.parallel_backend("dask"):
                rf.fit(train_features, train_labels)
            #rf.fit(train_features, train_labels)
            predictions = rf.predict(test_features)
            errors = np.absolute(predictions-test_labels)
            _mae = np.mean(errors)
            avg_tree_err+=_mae
            print('Mean Absolute Error: ${:7,.2f}'.format(_mae))
            mape = 100 * (errors / test_labels)
            accuracy = 100 - np.mean(mape)
            csv_out.writerow([ int(2**num_trees), trial, _mae, accuracy ])
            out.flush()
        avg_tree_err/=N_PER_TREE_CHOICE
        print("\n", "#"*20, "\n\nAverage error for {:4d} trees over {:3d} trials is ${:7,.2f}, {:5.2f} minutes elapsed\n\n".format(int(2**num_trees), N_PER_TREE_CHOICE, avg_tree_err, (time.time()-t0)/60.0), "#"*20, "\n")
