import gurobipy as gp
from gurobipy import Model, GRB, GurobiError, LinExpr, QuadExpr, GenExpr, abs_
import random
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import time
import json
from copy import deepcopy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] # set root directory
sys.path.append(os.path.abspath(ROOT))

from utils.flexibility_prediction.potential_policy import potential_policy,potential_opt
from utils.flexibility_prediction.baseline_policy import baseline_policy
from utils.system_estimate import traveling_time_est, trans_prob_manipulation, occupied_taxi_estimate
from utils.taxi import curise_region

""" Function: recover miscalculated idle driving distance and idle waiting time from MPC simulation result
"""
def idle_recovery(result_folder:Path,model,horizon=None,save=False):
    with open(result_folder.joinpath("parameter_constant.json"),"r") as f:
        setting = json.load(f)
        parameter = setting["parameter"]
        constant = setting["constant"]
        date = setting["data date"]

    with open(result_folder.joinpath("result.pickle"),"rb") as f:
        result = pickle.load(f)
        taxis = result["taxi_state"][0]
        decision_overall = result["decision"]
        metric_overall = result["metric"]


    #region: preprocess parameters
    # Hyper-parameters
    if "hyperParameter" in parameter.keys():
        low_initial_battery = parameter["hyperParameter"]["low_initial_battery"] # lower bound of initial battery level
        high_initial_battery = parameter["hyperParameter"]["high_initial_battery"] # upper bound of initial battery level
        max_occupied_length = parameter["hyperParameter"]["high_initial_battery"]  # maximum trip length of occupied taxis, unit: 10 min
    else:
        low_initial_battery = 5 # lower bound of initial battery level
        high_initial_battery = constant["L"] # upper bound of initial battery level
        max_occupied_length = 10  # maximum trip length of occupied taxis, unit: 10 min
    
    data_path = os.path.join(ROOT,'data','taxitripperslot',date)

    parameter_ori = parameter.copy()
    parameter = parameter.copy() # dict is passed by reference
    parameter['t_start'] = int(np.floor(parameter['t_start']/10))
    parameter['t_hat'] = int(np.floor(parameter['t_hat']/10))
    parameter['T_b'] = int(np.floor(parameter['T_b']/10))
    parameter['tau'] = int(np.floor(parameter['tau']/10))
    parameter['T_c'] = int(np.floor(parameter['T_c']/10)) # length of control horizon for each optimization problem
    parameter['horizon'] = int(np.floor(parameter['horizon']/10)) # lengh of total simulation
    if "slot_length" in parameter.keys():
        parameter["slot_length"] = int(np.floor(parameter['slot_length']/10))
    else:
        parameter["slot_length"] = 1 # 10 min

    if horizon is None:
        horizon = parameter["horizon"]

    if model == "TRC":
        # horizon = int(np.ceil(horizon/parameter["slot_length"]/10))
        horizon = int(np.ceil(horizon/parameter["slot_length"]))
    else:
        # horizon = int(np.ceil(horizon/parameter["slot_length"]/10)) # number of horizon
        horizon = int(np.ceil(horizon/parameter["slot_length"])) # number of horizon

    L_cap = 174 # battery capacity, unit: mile
    # L_increase = L_cap*10/48.0
    L_increase = L_cap*constant["L2"]/constant["L"]
    L_decrease = 7.0
    # L_threshold = L_cap*0.15
    L_threshold = L_cap*constant["L_threshold"]/constant["L"]
    idle_dis_dt = 2.5 * parameter["slot_length"] # straight line distance of idle runing in a time slot
    ratio = 0.5 # ratio between straight line distance and actual driving distance
    #endregion

    #region: load static data
    points_per_region=[] # ???: number of charging slot in each region
    fopen = open(ROOT.joinpath("data","pointsnum_per_region"),'r')
    for k in fopen:
        k=k.strip('\n')
        points_per_region.append(int(float(k)))
    fopen.close()

    # traveling distance
    travelingdistance_region_to_region = []
    with open(ROOT.joinpath("data","travelingdistance_region")) as f:
        for line in f:
            line= line.strip('\n')
            line = line.split(',')
            one=[]
            for k in line:
                one.append(float(k))
            travelingdistance_region_to_region.append(one)
    
    # load reachlist data
    reachlist=[]
    for region in range(constant["num_of_regions"]):
        tmp, one = traveling_time_est(region,constant=constant,slot_length=parameter["slot_length"])
        reachlist.append(one)

    # c=[]
    # fopen = open(ROOT.joinpath("data", 'reachable'),'r')
    # for k in fopen:
    #     k=k.strip('\n')
    #     k=k.split(',')
    #     one =[]
    #     for value in k:
    #         one.append(int(float(value)))
    #     c.append(one)
    # fopen.close()
    # for region in range(constant["num_of_regions"]):
    #     one=[]
    #     for j in range(constant["num_of_regions"]):
    #         if c[region][j]==0:
    #             one.append(j)
    #     reachlist.append(one)

    #endregion

    #region: Simulation
    J_trans = np.zeros((constant['num_of_regions'],horizon))
    num_charging = np.zeros((constant['num_of_regions'],constant['L'],horizon))
    J_idle = np.zeros(horizon)          # idle driving distance
    J_idle_waiting = np.zeros(horizon)  # idle waiting time

    # decision_overall = {
    #     "xp":np.zeros((constant['num_of_regions'],constant['num_of_regions'],constant['L'],horizon)),
    #     "xd":np.zeros((constant['num_of_regions'],constant['num_of_regions'],constant['L'],horizon)),
    #     "yp":np.zeros((constant['num_of_regions'],constant['num_of_regions'],constant['L'],horizon)),
    #     "yd":np.zeros((constant['num_of_regions'],constant['num_of_regions'],constant['L'],horizon)),
    #     "u":np.zeros((constant['num_of_regions'],constant['L'],horizon))
    # }

    state_overall = {
        "vacant":np.zeros((constant['num_of_regions'],constant['L'],horizon)),
        "charging":np.zeros((constant['num_of_regions'],constant['L'],horizon)),
        "waiting":np.zeros((constant['num_of_regions'],constant['L'],horizon)),
        "occupied":np.zeros((constant['num_of_regions'],constant['L'],max_occupied_length,horizon)),
        "estimated_demand":np.zeros((constant['num_of_regions'],horizon)),
        "deltaP_opt": np.zeros(horizon), # Delta P value from optimization solution
        "J_trans_opt":np.zeros((constant['num_of_regions'],horizon))
    }

    counter = 0 # number of steps excuted in current decision

    tau = int(parameter["tau"]/parameter["slot_length"])
    for t in range(tau):
        time = t*parameter["slot_length"]+parameter["t_start"]+parameter["t_hat"]

        # print('********************************* ', time, " ******************************************")

        if model == "TRC":
            t_tmp = int(t+parameter["t_hat"]/parameter["slot_length"])
            taxis = result["taxi_state"][t_tmp].copy()
        else:
            taxis = result["taxi_state"][t].copy()

        taxis["remaining_energy"] = taxis["remaining_energy"].astype("float")

        charging_slot_free = points_per_region.copy()

        #region: load real-time dynamic data
        # load demand data at current time slot
        current_demand=np.zeros((constant["num_of_regions"],constant["num_of_regions"]))
        for t1 in range(parameter["slot_length"]):
            tmp = []
            with open(os.path.join(data_path,str(time+t1)+'_demand'),'r') as f:
                for line in f:
                    line = line.strip('\n')
                    line = line.split(',')
                    one=[]
                    for k in line:
                        one.append(float(k))
                    tmp.append(one)
            current_demand += np.array(tmp)

        state_overall["estimated_demand"][:,t] = np.sum(current_demand,1)

        # load trip data at current time slot
        current_trips=[]
        for i in range(constant['num_of_regions']):
            current_trips.append([])
        for t1 in range(parameter["slot_length"]):
            with open(os.path.join(data_path,str(time+t1)+'_trips'),'r') as f:
                for line in f:
                    line = line.strip('\n')
                    line = line.split(',')
                    one = [int(float(line[0])), int(float(line[1])), int(float(line[2])), int(float(line[3])), float(line[4]),
                        float(line[5]), float(line[6]), float(line[7]), float(line[8]),float(line[9])]
                    current_trips[one[2]].append(one)
                    # totalvalue[time-36] += float(line[9])
        
        # calculate fleet state with status of e-taxis
        for n in range(constant['num_of_taxis']):
            l = min(int(taxis.loc[n,"remaining_energy"]/(idle_dis_dt/ratio)),constant['L']-1) # calcuate energy level # type: ignore
            if taxis.loc[n,"status"] == 3: # the taxi is vacant
                i = taxis.loc[n,"location"] # type: ignore
                state_overall['vacant'][i,l,t] += 1
            elif taxis.loc[n,"status"] == 1: # the taxi is waiting
                i = taxis.loc[n,"location"] # type: ignore
                state_overall['waiting'][i,l,t] += 1
            elif taxis.loc[n,"status"] == 0: # the taxi is charging
                i = taxis.loc[n,"location"] # type: ignore
                state_overall['charging'][i,l,t] += 1
            elif taxis.loc[n,"status"] == 2: # the taxi is occupied
                i = taxis.loc[n,"destination"] # type: ignore
                arriving_time = int((taxis.loc[n,"arriving_time"]/10-time)/parameter["slot_length"])# type: ignore
                if arriving_time == 0:
                    state_overall["vacant"][i,l,t] += 1 # type: ignore
                elif arriving_time < max_occupied_length:
                    state_overall['occupied'][i,l,arriving_time,t] += 1 # type: ignore
        #endregion

        #region: derive coordination decisions   

        if model == "TRC":
            t_tmp = int(t+parameter["t_hat"]/parameter["slot_length"])
            decision = {
                "xp":decision_overall['xp'][:,:,:,t_tmp:],
                "xd":decision_overall['xd'][:,:,:,t_tmp:],
                "yp":decision_overall['yp'][:,:,:,t_tmp:],
                "yd":decision_overall['yd'][:,:,:,t_tmp:],
                "u":decision_overall['u'][:,:,t_tmp:]
            }

            num_served = metric_overall["J_trans"][:,t_tmp]
        else:
            decision = {
                "xp":decision_overall['xp'][:,:,:,t:],
                "xd":decision_overall['xd'][:,:,:,t:],
                "yp":decision_overall['yp'][:,:,:,t:],
                "yd":decision_overall['yd'][:,:,:,t:],
                "u":decision_overall['u'][:,:,t:]
            }

            num_served = metric_overall["J_trans"][:,t]

        counter = 0
        #endregion
        
        #region: apply coordination decisions
        taxis['update_status'] = 0 # none of the taxis has taken actions yet.

        #region: update existing trip
        for n in range(constant['num_of_taxis']):
            l = int(min(taxis.loc[n,"remaining_energy"]/(idle_dis_dt/ratio),constant['L']-1)) # type: ignore
            i = taxis.loc[n,"location"] # type: ignore
            if taxis.loc[n,"status"] == 2: # occupied
                if taxis.loc[n,"update_status"] == 0:
                    if taxis.loc[n,"arriving_time"]/10-time <=parameter["slot_length"]: # type: ignore
                        taxis.loc[n,"status"] = 3
                        taxis.loc[n,"location"] = taxis.loc[n,"destination"]
                        taxis.loc[n,"update_status"] = 0
                    else:
                        taxis.loc[n,"status"] = 2
                        taxis.loc[n,"update_status"] = 1
        #endregion

        #region: update taxi states according to control decisions
        for n in range(constant['num_of_taxis']):
            l = int(min(taxis.loc[n,"remaining_energy"]/(idle_dis_dt/ratio),constant['L']-1)) # type: ignore
            i:int = taxis.loc[n,"location"] # type: ignore
            if taxis.loc[n,"status"] == 3: # unoccupied
                find = False # flag for whether action is assigned
                for j in reachlist[i]:
                    if int (decision['xp'][i,j,l,counter])>0:# follow xp decision: from vacant to charging station in adjacent regions (including current region)
                        if j != i:
                            taxis.loc[n,"remaining_energy"] -= travelingdistance_region_to_region[i][j]/ratio
                            J_idle[t] += travelingdistance_region_to_region[i][j]
                        else:
                            taxis.loc[n,"remaining_energy"] -= 1 # type: ignore
                            J_idle[t] += 1.0
                        taxis.loc[n,"remaining_energy"] = max(0,taxis.loc[n,"remaining_energy"]) # type: ignore
                        taxis.loc[n,"location"] = j
                        taxis.loc[n,"destination"] = j
                        taxis.loc[n,'status'] = 4

                        decision["xp"][i,j,l,counter] -= 1

                        find = True
                        break
                if not find: 
                    for j in reachlist[i]:
                        if int(decision["yp"][i,j,l,counter])>0:# follow yp decision: from vacant to vacant in adjacent regions (including current region)
                            if j != i:
                                taxis.loc[n,"remaining_energy"] -= travelingdistance_region_to_region[i][j]/ratio # TODO energy consumption of cruise
                                J_idle[t] += travelingdistance_region_to_region[i][j]
                            else:
                                taxis.loc[n,"remaining_energy"] -= 1 # type: ignore
                                J_idle[t] += 1
                            taxis.loc[n,"remaining_energy"] = max(0,taxis.loc[n,"remaining_energy"]) # type: ignore
                            taxis.loc[n,"location"] = j
                            taxis.loc[n,'status'] = 3
                            # taxis.loc[n,"update_status"] = 1 # dispatched to vacant taxis are not able to serve passengers in this time slot
                            
                            decision["yp"][i,j,l,counter] -= 1

                            find = True
                            break
                if not find and taxis.loc[n,"remaining_energy"] < L_threshold+1: # type: ignore
                    taxis.loc[n,"status"] = 4
                    taxis.loc[n,"destination"] = i
                    taxis.loc[n,"remaining_energy"] -= 0.5 # type: ignore
                if not find and taxis.loc[n,"remaining_energy"] >= L_threshold+1: # type: ignore
                    taxis.loc[n,"status"] = 3
                    # taxis.loc[n,"update_status"] = 1 # dispatched to vacant taxis are not able to serve passengers in this time slot
            elif taxis.loc[n,"status"] in [0,1]: # charging or waiting
                find = False # flag for whether action is assigned
                for j in reachlist[i]:
                    if int (decision['xd'][i,j,l,counter])>0: # follow xd decision: from charging station to charging station in adjacent regions (including current region)
                        if j != i:
                            taxis.loc[n,"remaining_energy"] -= travelingdistance_region_to_region[i][j]/ratio # type: ignore
                            J_idle[t] += travelingdistance_region_to_region[i][j]
                        else:
                            taxis.loc[n,"remaining_energy"] -= 1 # type: ignore
                            J_idle[t] += 1
                        taxis.loc[n,"remaining_energy"] = max(0,taxis.loc[n,"remaining_energy"]) # type: ignore
                        taxis.loc[n,"location"] = j
                        taxis.loc[n,"destination"] = j
                        taxis.loc[n,'status'] = 4

                        decision["xd"][i,j,l,counter] -= 1

                        find = True
                        break
                if not find: 
                    for j in reachlist[i]:
                        if int(decision["yd"][i,j,l,counter])>0:# follow yd decision: from charging station to vacant in adjacent regions (including current region)
                            if j != i:
                                taxis.loc[n,"remaining_energy"] -= travelingdistance_region_to_region[i][j]/ratio # type: ignore
                                J_idle[t] += travelingdistance_region_to_region[i][j]
                            else:
                                taxis.loc[n,"remaining_energy"] -= 1 # type: ignore
                                J_idle[t] += 1
                            taxis.loc[n,"remaining_energy"] = max(0,taxis.loc[n,"remaining_energy"]) # type: ignore
                            taxis.loc[n,"location"] = j
                            taxis.loc[n,'status'] = 3
                            # taxis.loc[n,"update_status"] = 1 # dispatched to vacant taxis are not able to serve passengers in this time slot
                            
                            decision["yd"][i,j,l,counter] -= 1

                            find = True
                            break
                if not find and taxis.loc[n,"remaining_energy"] < L_threshold+1: # type: ignore
                    taxis.loc[n,"status"] = 4
                    taxis.loc[n,"destination"] = i
                    taxis.loc[n,"remaining_energy"] -= 0.5 # type: ignore
                if not find and taxis.loc[n,"remaining_energy"] >= L_threshold+1: # type: ignore
                    taxis.loc[n,"status"] = 3
                    # taxis.loc[n,"update_status"] = 1 # dispatched to vacant taxis are not able to serve passengers in this time slot
            
            i = taxis.loc[n,"location"] # type: ignore
            if taxis.loc[n,"status"] == 4: # dispatched to charging station
                if charging_slot_free[i] > 0 and int(decision["u"][i,l,counter]) > 0: # get charged
                    taxis.loc[n,"remaining_energy"] += L_increase # type: ignore
                    taxis.loc[n,"status"] = 0
                    taxis.loc[n,"update_status"] = 1
                    charging_slot_free[i] -= 1
                    decision["u"][i,l,counter] -= 1

                    num_charging[i,l,t] += 1
                else: # wait
                    taxis.loc[n,"status"] = 1
                    taxis.loc[n,"update_status"] = 1
                    J_idle_waiting[t] += 1
        #endregion

        #region: update taxi state according to trip data
        for n in range(constant['num_of_taxis']):
            if taxis.loc[n,"status"] == 3: # taxi is not occupied
                l = int(min(taxis.loc[n,"remaining_energy"]/(idle_dis_dt/ratio),constant['L']-1)) # type: ignore
                i = taxis.loc[n,"location"] # type: ignore
                if len(current_trips[i])>0 and num_served[i]>=1: # have trip in region i
                    trip_id = np.random.randint(0,len(current_trips[i]))
                    trip = current_trips[i][trip_id]
                    current_trips[i].remove(trip)
                    J_trans[i,t] += 1
                    num_served[i] -= 1
                    if trip[1]/10-time > parameter["slot_length"]: # the trip doesn't finish in current time slot
                        taxis.loc[n,"status"] = 2
                        taxis.loc[n,"destination"] = trip[3]
                        taxis.loc[n,"arriving_time"] = trip[1]
                        # remaining energy decrease by two parts: trip & cruise
                        taxis.loc[n,"remaining_energy"] -= trip[8]/ratio
                        # taxis.loc[n,"remaining_energy"] -= (idle_dis_dt / ratio) * ((int(trip[1] / 10) + 1) * 10 - (int(trip[0] / 10)) * 10
                        #                                              - (trip[1] - trip[0])) / 10.0
                        n_trip_slot = np.ceil((trip[1]/10-time)/parameter["slot_length"])
                        taxis.loc[n,"remaining_energy"] -= (idle_dis_dt / ratio)/(10*parameter["slot_length"]) * \
                            (((time+parameter["slot_length"]*n_trip_slot)*10-trip[1])+(trip[0]-time*10))
                        taxis.loc[n,"remaining_energy"] = max(0,taxis.loc[n,"remaining_energy"]) # type: ignore
                        
                        taxis.loc[n,"update_status"] = 1
                    else: # trip is finished in current time slot
                        taxis.loc[n,"status"] = 3
                        taxis.loc[n,"destination"] = trip[3]
                        taxis.loc[n,"location"] = trip[3]

                        taxis.loc[n,"remaining_energy"] -= trip[8]/ratio
                        taxis.loc[n,"remaining_energy"] -= (idle_dis_dt/ratio)*(1-(trip[1]-trip[0])/(10*parameter["slot_length"]))
                        taxis.loc[n,"remaining_energy"] = max(0,taxis.loc[n,"remaining_energy"]) # type: ignore
                        taxis.loc[n,"update_status"] = 1
                else: # no trip, taxi curise between regions
                    if model == "R2I":
                        if charging_slot_free[i] > 0: # R2I, idle taxis go charging
                            taxis.loc[n,"remaining_energy"] += L_increase # type: ignore
                            taxis.loc[n,"status"] = 0
                            taxis.loc[n,"update_status"] = 1
                            charging_slot_free[i] -= 1
                            decision["u"][i,l,counter] -= 1

                            num_charging[i,l,t] += 1
                        else: # R2I, idle taxis stay unmoved to save energy when charging capacity is full
                            taxis.loc[n,"status"] = 1
                            taxis.loc[n,"update_status"] = 1
                            J_idle_waiting[t] += 1  
                    else:
                        taxis.loc[n,"status"] = 3
                        j = curise_region(
                            time=time,
                            start_region=i,
                            travelingdistance_region_to_region=travelingdistance_region_to_region,
                            num_of_regions=constant['num_of_regions'],
                            idle_dis_dt=idle_dis_dt)
                        taxis.loc[n,"remaining_energy"] -= idle_dis_dt / ratio # type: ignore
                        taxis.loc[n,"location"] = j
                        taxis.loc[n,"update_status"] = 1

                        # J_idle[t] += idle_dis_dt / ratio

            """elif taxis.loc[n,"status"] == 2: # occupied
                if taxis.loc[n,"update_status"] == 0:
                    if int(taxis.loc[n,"arriving_time"]/10) == time:
                        taxis.loc[n,"status"] = 3
                        taxis.loc[n,"location"] = taxis.loc[n,"destination"]
                        taxis.loc[n,"update_status"] = 1
                    else:
                        taxis.loc[n,"status"] = 2
                        taxis.loc[n,"update_status"] = 1"""
        #endregion
        #endregion
    
    #endregion

    if save:
        setting["cost_estimation"] = {
            "idle_driving_provision":np.sum(J_idle),
            "idle_wating_provision":np.sum(J_idle_waiting)
        }

        setting_str = json.dumps(setting, indent=4)

        with open(result_folder.joinpath("parameter_constant.json"),"w") as f:
            f.write(setting_str)
    

    return J_idle, J_idle_waiting

""" The response policy of flexibility estimation in energy_ref group experiment are mistakenly calculated. The result can copied from TRC_ref group to fix it.
    The functionality of this function is to copy the corresponding response policy in flexibility estimation from TRC_ref to energy_ref group.
"""
def main_flexibility_est_res_revcovery():
    model = "RES"
    TRC_ref_path = ROOT.joinpath("results/final_MPC_results(trc_ref)")
    energy_ref_path = ROOT.joinpath("results/final_MPC_results(energy_ref)")

    cases = ["4","5","9","10","12","13","14","15","16","17"]
    trails = {
        # "1":"1",
        "2":"2",
        # "3":"3",
        # "4":"4",
        # "6":"6"
    } # {energy_ref_trail_No: trc_ref_trail_No}

    energy_ref_dirs = {}
    for dir in os.listdir(energy_ref_path.joinpath(model)):
        exp_no = dir.split("-")[0]
        cn = exp_no.split("_")[0]
        tn = exp_no.split("_")[1]

        if cn in cases and tn in trails.keys():
            energy_ref_dirs[(cn,tn)] = deepcopy(dir)

    trc_ref_dirs = {}
    for dir in os.listdir(TRC_ref_path.joinpath(model)):
        exp_no = dir.split("-")[0]
        cn = exp_no.split("_")[0]
        tn = exp_no.split("_")[1]

        if cn in cases and tn in trails.values():
            trc_ref_dirs[(cn,tn)] = deepcopy(dir)

    for (cn,tn) in energy_ref_dirs:
        energy_ref_dir = energy_ref_path.joinpath(model,energy_ref_dirs[(cn,tn)])
        trc_ref_dir = TRC_ref_path.joinpath(model,trc_ref_dirs[(cn,trails[tn])])

        print("energy expr path:\t",energy_ref_dir)
        print("TRC expr path:\t\t", trc_ref_dir)
        print()

        ## correct e_res and flexibility calculation
        with open(trc_ref_dir.joinpath("parameter_constant.json"),"r") as f:
            e_res = json.load(f)["flexibility_estimation"]["e_res"]

        with open(energy_ref_dir.joinpath("parameter_constant.json"),"r") as f:
            setting = json.load(f)

        setting["flexibility_estimation"]["e_res"] = e_res
        flexibility = setting["flexibility_estimation"]["e_ref"] - e_res
        setting["flexibility_estimation"]["flexibility"] = flexibility
        
        setting_str = json.dumps(setting,indent=4)
        with open(energy_ref_dir.joinpath("parameter_constant.json"),"w") as f:
            f.write(setting_str)


        ## correct simulation result
        with open(trc_ref_dir.joinpath("flexibility_evaluation.pickle"),"rb") as f:
            trc_result = pickle.load(f)

        with open(energy_ref_dir.joinpath("flexibility_evaluation.pickle"),"rb") as f:
            energy_result = pickle.load(f)

        energy_result["res_simulation"] = trc_result["res_simulation"]
        energy_result["flexibility"]["e_res"] = e_res
        energy_result["flexibility"]["flexibility"] = flexibility

        with open(energy_ref_dir.joinpath("flexibility_evaluation.pickle"),"wb") as f:
            pickle.dump(energy_result,f)

        pass

if __name__ == '__main__':
    
    """
    model = "PRO"
    root_path = ROOT.joinpath("results")
    result_folders = os.listdir(root_path.joinpath(model))
    trial_no = ["1"]
    cases = ["4","5","9","10","12","13","14","15"]
    dirs = []
    for dir in result_folders:
        exp_no = dir.split("-")[0]
        cn = exp_no.split("_")[0]
        tn = exp_no.split("_")[1]
        if tn in trial_no and cn in cases:
            dirs.append(dir)
    result_folders = dirs

    for dir in result_folders:
        t_start = time.time()
        idle_recovery(
            result_folder=root_path.joinpath(model,dir),
            model=model,
            horizon=60,
            save=True
        )
        t_end = time.time()
        print("Experiment: %s, runtime: %f"%(dir, t_end-t_start))
    # """

    # main_flexibility_est_res_revcovery()
    
    result_path = ROOT.joinpath("results","Oracle","0_1")
    model="Oracle"
    idle_recovery(result_folder=result_path,model=model,save=True)