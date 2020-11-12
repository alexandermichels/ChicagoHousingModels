import csv, datetime, joblib, time, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from dask.distributed import Client, progress


def rtr(train_features, train_labels, test_features, test_labels, ntree, height):
    rf = RandomForestRegressor(n_estimators = ntree, max_depth=height)
    with joblib.parallel_backend("dask"):
         rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    errors = np.absolute(predictions-test_labels)
    mae = np.mean(errors)
    print('Mean Absolute Error: ${:7,.2f}'.format(mae))
    mape = 100 * (errors / test_labels)
    return mae, np.mean(mape)

N_PER_TREE_CHOICE = 10
MIN_TREES = 1
MAX_TREES = 10

MIN_HEIGHT = 1
MAX_HEIGHT = 10

ml_df = pd.read_pickle("./data/Cleaned_Chicago_Sales.pkl")
label_cols = ["Sale Price"]
ml_df = ml_df.drop(["Estimate (Land)", "Estimate (Building)"], axis=1)

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

output_file = "./output/RFR-{}trials-[{}-{}trees]-[{}-{}height]-{}.csv".format(N_PER_TREE_CHOICE,int(2**MIN_TREES), int(2**(MAX_TREES)), MIN_HEIGHT, MAX_HEIGHT, datetime.datetime.now())

with Client(processes=False, threads_per_worker=12, n_workers=1, memory_limit='30GB') as client:
    with open(output_file, "w") as out:
        csv_out = csv.writer(out)
        csv_out.writerow([ "NTree", "MaxDepth", "Trial", "MAE", "MAPE"])
        for height in range(MIN_HEIGHT, MAX_HEIGHT+1):
            for num_trees in range(MIN_TREES,MAX_TREES+1):
                t0 = time.time()
                avg_tree_err = 0
                for trial in range(N_PER_TREE_CHOICE):
                    ntree = int(2**num_trees)
                    print("NTree: {:4d}, Trial {:3d}".format(ntree, trial))
                    mae, mape = rtr(train_features, train_labels, test_features, test_labels, ntree, height)
                    avg_tree_err+=mae
                    csv_out.writerow([ ntree, height, trial, mae, mape ])
                    out.flush()
                avg_tree_err/=N_PER_TREE_CHOICE
                print("\n", "#"*20, "\n\nAverage error for {:4d} trees with max depth {:2d} over {:3d} trials is ${:7,.2f}, {:5.2f} minutes elapsed\n\n".format(int(2**num_trees), height, N_PER_TREE_CHOICE, avg_tree_err, (time.time()-t0)/60.0), "#"*20, "\n")
