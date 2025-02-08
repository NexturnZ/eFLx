import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] # set root directory
sys.path.append(os.path.abspath(ROOT))


if __name__ == '__main__':
    data_path = ROOT.joinpath('data/taxitripperslot')

    num_of_regions = 38
    traveling_time = np.zeros((num_of_regions,num_of_regions))
    num_data = np.zeros((num_of_regions,num_of_regions))

    col_names = ["start_time","end_time","start_region","end_region"]
    for dir in os.listdir(data_path):
        dir_tmp = Path.joinpath(data_path,dir)
        for file in os.listdir(dir_tmp):
            if "trips" in file:
                data = pd.read_csv(Path.joinpath(dir_tmp,file),names=col_names,usecols=[0,1,2,3])
                for i in range(num_of_regions):
                    for j in range(num_of_regions):
                        tmp = data.query("start_region == @i and end_region == @j")
                        if len(tmp)>0:
                            time = tmp["end_time"]-tmp["start_time"]
                            time = time[time>0]

                            mean_time = np.mean(time)

                            traveling_time[i,j] = (traveling_time[i,j]*num_data[i,j] + mean_time*len(time))/(num_data[i,j]+len(time))
                            num_data[i,j] += len(time)

    
    traveling_timeslot = pd.DataFrame(np.ceil(traveling_time/10))
    traveling_timeslot = traveling_timeslot.fillna(0)

    result_path = ROOT.joinpath('data/travelingtime_region')
    traveling_timeslot.to_csv(result_path,sep=',',header=False,index=False)
    
    pass

