import numpy as np
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] # set root directory
sys.path.append(os.path.abspath(ROOT))

""" Function: calculate number of occupied taxis based on supply stored in baseline policy
max_occupied_length: maximum number of traveling time of occupied taxis stored;
t_hat: The number of time slot from the start  
"""
def occupied_taxi_estimate(parameter:dict,supply,t_hat,initial_occupied_taxis=None,constant=None,date="2015-01-01",max_occupied_length=10):
    parameter = parameter.copy()
    data_path = os.path.join(ROOT, 'data')
    demand_data_path = os.path.join(data_path,'taxitripperslot',date)

    if constant is None:
        constant = {
            "L":30,
            "L2":6,
            "L_threshold":4, # 4
            "num_of_taxis":13000, # 13000
            "num_of_regions":38
        }

    t_start = int(min(1440,parameter["t_start"])/10)
    slot_length = int(parameter["slot_length"]/10)

    #region: load matrix of traveling time
    reachlist=[]
    travelingTime = np.zeros((constant["num_of_regions"],constant["num_of_regions"]),dtype="int")
    for region in range(constant["num_of_regions"]):
        travelingTime[region,:], one = traveling_time_est(region,constant=constant,slot_length=slot_length)
        reachlist.append(one)

    # load tensor of demand
    transition_new = np.zeros((t_hat,constant["num_of_regions"],constant["num_of_regions"])) # transition matrix using demand data
    estimatedpassengerdemand = np.zeros((constant["num_of_regions"],t_hat)) # estimated passenger demand
    for slot in range(t_hat):

        current_demand=np.zeros((constant["num_of_regions"],constant["num_of_regions"]))
        for slot_sub in range(slot_length):
            tmp = []
            with open(os.path.join(demand_data_path,str(t_start+int(slot*slot_length)+slot_sub)+'_demand'),'r') as f:
                for line in f:
                    line = line.strip('\n')
                    line = line.split(',')
                    one=[]
                    for k in line:
                        one.append(float(k))
                    tmp.append(one)

            current_demand += np.array(tmp)
            
        estimatedpassengerdemand[:,slot] += np.sum(current_demand,1)

        transition_new[slot,:,:] = current_demand/np.reshape(np.sum(current_demand,1),(constant["num_of_regions"],1))  

    transition_new = np.nan_to_num(transition_new)

    transition_man = np.zeros((transition_new.shape[0],constant["num_of_regions"],constant["num_of_regions"],constant["L"]))
    for t in range(transition_man.shape[0]):
        transition_man[t,:,:,:] = trans_prob_manipulation(transition=transition_new[t,:,:],travelingTime=travelingTime,constant=constant)
    transition_new = transition_man

    #endregion


    if type(supply) == dict: # if input is Gurobi expressions
        occupied = {}
        for i in range(constant["num_of_regions"]): # departure 
            for l in range(constant["L"]):
                for t in range(max_occupied_length):
                    occupied[i,l,t] = 0
            

    else: 
        occupied = np.zeros((constant["num_of_regions"],constant["L"],max_occupied_length))

    for t in range(t_hat):
        for i in range(constant["num_of_regions"]): # departure 
            for j in range(constant["num_of_regions"]):
                for l in range(constant["L"]):
                    if j not in reachlist[i] and transition_new[t,i,j,l]>0:
                        travel_time = travelingTime[i,j] - (t_hat-t)
                        # if travel_time < occupied.shape[2]:
                        if travel_time >= 0 and travel_time < max_occupied_length:
                            if l >= travelingTime[i,j]:
                                occupied[j,l-travelingTime[i,j],travel_time] += supply[i,l,t]*transition_new[t,i,j,l]
                            else:
                                occupied[j,0,travel_time] += supply[i,l,t]*transition_new[t,i,j,l]
    
    """
    if initial_occupied_taxis is not None: # add initial occupied taxis
        if initial_occupied_taxis.shape[2] >= max_occupied_length:
            occupied[:,:,:max_occupied_length-t_hat] += initial_occupied_taxis[:,:,t_hat:max_occupied_length]
        else:
            w = initial_occupied_taxis.shape[2]
            occupied[:,:,:w-t_hat] += initial_occupied_taxis[:,:,t_hat:]
    """

    if initial_occupied_taxis is not None: # add initial occupied taxis
        w = min(initial_occupied_taxis.shape[2], max_occupied_length)
        for i in range(constant["num_of_regions"]):
            for l in range(constant["L"]):
                for t in range(w-t_hat):
                    occupied[i,l,t] += initial_occupied_taxis[i,l,t+t_hat]

    # occupied = np.round(occupied)

    return occupied

"""
slot_length: length of time slot, unit: 10min
"""
def traveling_time_est(initial_region,constant=None,slot_length=1):
    if constant is None:
        constant = {"num_of_regions":38}

    file_path = ROOT.joinpath("data","travelingdistance_region")
    travelingdistance_region_to_region = []
    with open(file_path) as f:
        for line in f:
            line = line.strip('\n')
            line = line.split(',')
            one = []
            for k in line:
                one.append(float(k))
            travelingdistance_region_to_region.append(one)


    reachable = []
    traveling_time = np.zeros(constant["num_of_regions"],dtype='int')
    for i in range(constant["num_of_regions"]):
        traveling_time[i] = int(np.ceil(travelingdistance_region_to_region[initial_region][i] / (2.5*slot_length)))
        if traveling_time[i] <= 1:
            reachable.append(i)

    # print (potential_region)
    return traveling_time, reachable

""" TODO: output description

"""
def trans_prob_manipulation(transition,travelingTime,constant):
    transition_new = np.zeros((constant["num_of_regions"],constant["num_of_regions"],constant["L"]))
    for i in range(constant["num_of_regions"]):
        for l in range(constant["L"]):
            A = (travelingTime[i] > l) # unreachable set
            Ac = (travelingTime[i] <= l) # reachable set
            transition_new[i,A,l] = 0
            transition_new[i,Ac,l] = transition[i,Ac] + np.sum(transition[i,A])/np.sum(Ac)
    
    return transition_new

if __name__ == '__main__':
    traveling_time, reachlist = traveling_time_est(34,slot_length=1)
    print(reachlist)
    print(traveling_time)

