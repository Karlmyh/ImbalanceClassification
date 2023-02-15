import os
import numpy as np 
import pandas as pd
import glob





# real data tree summarize
log_file_dir = "./realdataresult"


method_seq = glob.glob("{}/*.csv".format(log_file_dir))
method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)
summarize_log=pd.DataFrame([])

for method in method_seq:
    log = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    log.columns = "dataset,iteration,time,am,n1,n2".split(',')
    log["method"]=method
    summarize_log=summarize_log.append(log)
    
 
summary = pd.pivot_table(summarize_log, index=["dataset"],columns=["method"], values=[ "am","time"], aggfunc=[np.mean, np.std, len])
summary.to_excel("./realdata_summary.xlsx")

