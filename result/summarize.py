import os
import numpy as np 
import pandas as pd
import glob


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


# real data tree summarize
log_file_dir = "./realdataresult"


method_seq = glob.glob("{}/*.csv".format(log_file_dir))
method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)
summarize_log=pd.DataFrame([])

all_results_seq = []
for method in method_seq:
    if method == "BorderlineSMOTE":
        continue
    print(method)
    logs = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    logs.columns = ["dataset_name", "idx", "n_neighbors1", "n_neighbors2", "random_state", "time_exp", "test_acc", "test_recall", "test_precision", "test_f1", "test_gmean"]
    logs = logs.sort_values(["dataset_name", "idx", "n_neighbors1", "n_neighbors2"])
    logs = logs.reset_index(drop=True)
    logs = logs.drop(["random_state", "test_acc", "test_precision", "test_f1"], axis=1)
    best_idx = logs.groupby(["dataset_name", "idx"]).idxmax()["test_recall"]
    
    best_idx[best_idx.isna()] = 0
        
    results = logs.loc[best_idx]
    results = results.groupby(["dataset_name"]).agg({
            "test_recall": ["mean", "std"],
            "test_gmean": ["mean", "std"],
            "time_exp": ["mean", "std", len]
        })
    results.columns = ['_'.join(col) for col in results.columns.values]
    results = results.reset_index()
    results = results.sort_values(["dataset_name"])
    results.insert(1, "method", method)
    results = results.reset_index(drop=True)
    all_results_seq.append(results)
all_results = pd.concat(all_results_seq, axis=0)
all_results = all_results.reset_index(drop=True)
all_results = all_results.sort_values(['dataset_name', 'method'])
all_results = all_results.set_index(keys=['dataset_name', 'method'])
print(all_results)
all_results.to_excel("./realdata_summary.xlsx", index=True)








