import gurobipy as gp
from gurobipy import Model, GRB, GurobiError, LinExpr, QuadExpr, GenExpr, abs_
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import time as Time
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] # set root directory
sys.path.append(os.path.abspath(ROOT))

from utils.flexibility_prediction.potential_policy import potential_policy,potential_opt
from utils.flexibility_prediction.baseline_policy import baseline_policy
from utils.system_estimate import traveling_time_est, trans_prob_manipulation, occupied_taxi_estimate
from utils.taxi import curise_region

""" Function: iterative algorithm for flexibility provisioning.
    Input:
        1. parameter:
            a. t_start: simulation start time (unit:min)
            b. t_hat: request start time hat{t} (e.g. t_hat=20 implies the request start 20 minutes after t_start)
            c. T_b: request duration
            d. tau: ramping duration
            e. T_c: control horizon
            f. epsilon: tolerance on decrease transportation performance
            g. ita: tolerance on increase idle driving distance
            h. max_iter: number of maximum iteration
        2. constant=[num_of_regions,num_of_taxis,L,L2,L_threshold]:
        3. J_trans: total number of passenger served of baseline policy
        4. J_idle: total idle driving distance of baseline policy
        5. P: charging power of baseline policy
        6. initial_state: initial {vacant, waiting, charging, occupied}
        7. date: data for which date is used
    Output:
        1. initial_solution[decision, metric, state]: solution with external constraints (type: dict)
        2. final_solution[decision,metric_state]: solution with calibrated constraints (type: dict)
"""
def provision_model(parameter:dict, constant:dict, J_trans:np.ndarray, J_idle:np.ndarray, P:np.ndarray, initial_state:dict, date:str='2015-01-01',log:bool=False):
    parameter = parameter.copy()

    decision_first, metrics_first, state_first = potential_policy(
        parameter=parameter,
        constant=constant,
        J_trans=J_trans,
        J_idle=J_idle,
        P=P,
        initial_state=initial_state,
        date=date,
        log=log)

    metrics_old = metrics_first
    
    first_solution = {"decision":decision_first,"metric":metrics_first,"state":state_first}

    if parameter["use_first_sol"]:
        return first_solution, None

    tau = int(np.floor(parameter["tau"]/parameter["slot_length"]))

    occupied = occupied_taxi_estimate(
        parameter=parameter.copy(),
        supply=state_first["supply"], # type: ignore
        t_hat=tau,
        initial_occupied_taxis=initial_state["occupied"],
        constant=constant,
        date=date
    )

    flex_state = {
        "vacant":state_first["vacant"][:,:,tau],    # type: ignore
        "charging":state_first["charging"][:,:,tau],# type: ignore
        "waiting":state_first["waiting"][:,:,tau],  # type: ignore
        "occupied":occupied
    }

    parameter_low = parameter.copy()
    parameter_low["t_start"] = parameter["t_start"] + parameter["tau"]
    parameter_low["T_c"] -= parameter['tau']
    parameter_low["tau"] = 0

    print("--num_charging: ",np.sum(metrics_old["num_charging"][:,:,tau:])) # type: ignore

    try:
        iter_flag = True    # initialize iteration flag
        num_iter = 0        # initialize iteration number
        while iter_flag:
            print("----Iteration %d----"%(num_iter))
            # evaluate new baseline
            print("--Evaluate baseline:")
            _, metric_ref_new ,state_ref_new ,_ = baseline_policy(
                parameter=parameter_low,
                constant=constant,
                initial_state=flex_state,
                date=date
            )

            J_trans_new = J_trans.copy()
            J_trans_new[:,tau:] = metric_ref_new["J_trans"]

            J_idle_new = J_idle.copy()
            J_idle_new[tau:] = metric_ref_new["J_idle"]

            P_new = P.copy()
            P_new[:,:,tau:] = metric_ref_new["num_charging"]

            # update solution
            print("\n--Update Provisioning solution:")
            decision_new, metrics_new, state_new = potential_policy(
                parameter=parameter,
                constant=constant,
                J_trans=J_trans_new,
                J_idle=J_idle_new,
                P=P_new,
                initial_state=initial_state,
                date=date,
                log=log
            )

            occupied = occupied_taxi_estimate(
                parameter=parameter.copy(),
                supply=state_first["supply"], # type: ignore
                t_hat=tau,
                initial_occupied_taxis=initial_state["occupied"],
                constant=constant,
                date=date
            )

            
            flex_state = {
                "vacant":state_first["vacant"][:,:,tau],    # type: ignore
                "charging":state_first["charging"][:,:,tau],# type: ignore
                "waiting":state_first["waiting"][:,:,tau],  # type: ignore
                "occupied":occupied
            }

            # estimate solution difference
            sol_diff = np.linalg.norm(metrics_new["num_charging"][:,:,tau:]-metrics_old["num_charging"][:,:,tau:]) # type: ignore

            # verify whether to continue iteration
            num_iter += 1 # update iteration number
            iter_flag = (num_iter<parameter["max_iter"]) and (sol_diff > parameter["sol_diff_th"])   # update iteration flag
            metrics_old = metrics_new

            print("-iteration: %d\n--num_charging: %f"%(num_iter,np.sum(metrics_old["num_charging"][:,:,tau:]))) # type: ignore

        final_solution = {"decision":decision_new,"metric":metrics_new,"state":state_new}
    except:
        final_solution = None

    return first_solution, final_solution

def provision_model_mpc(parameter:dict, constant:dict, J_trans_baseline:np.ndarray, J_idle_baseline:np.ndarray, P_baseline:np.ndarray, initial_state, date:str='2015-01-01',log:bool=False):
    #region: preprocess parameters
    if constant is None:
        constant = {
            "L":30,
            "L2":6,
            "L_threshold":4,
            "num_of_taxis":13000, # 13000
            "num_of_regions":38
        }
    
    # Hyper-parameters
    if "hyperParameter" in parameter.keys():
        low_initial_battery = parameter["hyperParameter"]["low_initial_battery"] # lower bound of initial battery level
        high_initial_battery = parameter["hyperParameter"]["high_initial_battery"] # upper bound of initial battery level
        max_occupied_length = parameter["hyperParameter"]["max_occupied_length"]  # maximum trip length of occupied taxis, unit: 10 min
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

    horizon = int(np.ceil(parameter["horizon"]/parameter["slot_length"])) # number of horizon

    L_cap = 174 # battery capacity, unit: mile
    # L_increase = L_cap*10/48.0
    L_increase = L_cap*constant["L2"]/constant["L"]
    L_decrease = 7.0
    # L_threshold = L_cap*0.15
    L_threshold = L_cap*constant["L_threshold"]/constant["L"]
    idle_dis_dt = 2.5 * parameter["slot_length"] # straight line distance of idle runing in a time slot
    ratio = 0.5 # ratio between straight line distance and actual driving distance
    #endregion


    #region: randomly generate initial state
    if initial_state is None:
        """
        location: integer in [0,38)
        status: 0: being charged, 1: waiting for a free charging point, 2: occupied, 3: unoccupied, 4: assigned to region for charging
        destination: integer in [-1,38), -1 represents not set.
        arriving_time: -1 represents not set.
        update_status: whether status has been updated: 0->no; 1-> yes;
        """
        taxis = pd.DataFrame(
            columns=["location","status","destination","arriving_time","remaining_energy","charging_status","update_status"],
            dtype="int",
            index=np.arange(constant['num_of_taxis'])
        )
        taxis['destination'] = -1
        taxis['arriving_time'] = -1
        taxis['charging_status'] = 0
        taxis['update_status'] = 1

        initial_demand=[]
        for i in range(constant["num_of_regions"]):
            one=[]
            for j in range(constant["num_of_regions"]):
                one.append(0.0)
            initial_demand.append(one)
        for slot in range(parameter['t_start']+parameter['t_hat']-1,parameter['t_start']+parameter['t_hat']+24):
            perslot=[]
            with open(os.path.join(data_path,str(slot)+'_demand')) as f:
                for line in f:
                    line = line.strip('\n')
                    line = line.split(',')
                    one=[]
                    for k in line:
                        one.append(int(float(k)))
                    perslot.append(one)
                for i in range(constant["num_of_regions"]):
                    for j in range(constant["num_of_regions"]):
                        initial_demand[i][j] += perslot[i][j]
            break
        for i in range(constant["num_of_regions"]):
            for j in range(constant["num_of_regions"]):
                initial_demand[i][j] = int(initial_demand[i][j]/1.0)

        initial_select=[] # initial_select will be [1,...,1, 2,...,2, ..., 35,...,35]
        for i in range(constant["num_of_regions"]):
            num = sum(initial_demand[i][j] for j in range(constant["num_of_regions"]))
            for j in range(num):
                initial_select.append(i)
        initial_select = np.array(initial_select)
        np.random.shuffle(initial_select)


        # load trip data 3 time slots before simulation to estimate initial taxi state
        data=[]
        for t in range(parameter['t_start']+parameter['t_hat']-3,parameter['t_start']+parameter['t_hat']-1):
            if t >= 35:
                file = os.path.join(data_path, str(t)+'_trips')
                with open(file) as f:
                    for line in f:
                        line = line.strip('\n')
                        line = line.split(',')
                        if int(line[1]) > (parameter['t_start'] + parameter['t_hat'])*10:
                            c = [int(float(line[0])),int(float(line[1])),int(float(line[2])),int(float(line[3]))]
                            data.append(c)
            else:
                break
        
        file = os.path.join(data_path, str(parameter['t_start']+parameter['t_hat']-1)+'_trips')
        with open(file) as f:
            for line in f:
                line = line.strip('\n')
                line = line.split(',')
                c = [int(float(line[0])),int(float(line[1])),int(float(line[2])),int(float(line[3]))]
                data.append(c)

        # initialize taxi location, status, destination and arriving_time with trip data
        for i in range(len(data)):
            line = data[i]
            if line[1]<(parameter['t_start']+parameter['t_hat'])*10: # if the taxi finishes its current service during the current slot, it will still be unoccupied next time slot
                taxis.loc[i,"location"] = line[3]
                taxis.loc[i,"status"] = 3
                taxis.loc[i,"destination"] = -1
                taxis.loc[i,"arriving_time"] = -1
            elif line[1]>=(parameter['t_start']+parameter['t_hat'])*10 and line[1] < (parameter['t_start']+parameter['t_hat']+1)*10:
                taxis.loc[i,"location"] = line[3]
                taxis.loc[i,"status"] = 2
                taxis.loc[i,"destination"] = line[3]
                taxis.loc[i,"arriving_time"] = line[1]
            else:
                taxis.loc[i,"location"] = line[3]
                taxis.loc[i,"status"] = 2
                taxis.loc[i,"destination"] = line[3]
                taxis.loc[i,"arriving_time"] = line[1]
        # remaining taxis are vacant
        for i in range(len(data),constant["num_of_taxis"]):
            taxis.loc[i,"location"] = np.random.choice(initial_select) # randomly arrange taxi location with fixed number of taxis in each region
            taxis.loc[i,"status"] = 3
            taxis.loc[i,"destination"] = -1
            taxis.loc[i,"arriving_time"] = -1
        taxis['remaining_energy'] = L_cap/constant["L"]*np.random.randint(
            low=low_initial_battery, # HYPERPARAMETER here
            high=high_initial_battery, # HYPERPARAMETER here
            size=constant["num_of_taxis"])
    else:
        taxis = initial_state.copy()

    taxis["remaining_energy"] = taxis["remaining_energy"].astype(float)
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
    J_idle = np.zeros(horizon)
    runtime = np.zeros(horizon)


    decision_overall = {
        "xp":np.zeros((constant['num_of_regions'],constant['num_of_regions'],constant['L'],horizon)),
        "xd":np.zeros((constant['num_of_regions'],constant['num_of_regions'],constant['L'],horizon)),
        "yp":np.zeros((constant['num_of_regions'],constant['num_of_regions'],constant['L'],horizon)),
        "yd":np.zeros((constant['num_of_regions'],constant['num_of_regions'],constant['L'],horizon)),
        "u":np.zeros((constant['num_of_regions'],constant['L'],horizon))
    }

    state_overall = {
        "vacant":np.zeros((constant['num_of_regions'],constant['L'],horizon)),
        "charging":np.zeros((constant['num_of_regions'],constant['L'],horizon)),
        "waiting":np.zeros((constant['num_of_regions'],constant['L'],horizon)),
        "occupied":np.zeros((constant['num_of_regions'],constant['L'],max_occupied_length,horizon)),
        "estimated_demand":np.zeros((constant['num_of_regions'],horizon)),
        "deltaP_opt": np.zeros(horizon), # Delta P value from optimization solution
        "J_trans_opt":np.zeros((constant['num_of_regions'],horizon))
    }

    taxi_state = [taxis.copy()] # detailed taxi state for each time slot
    counter = 0 # number of steps excuted in current decision

    tau = int(parameter["tau"]/parameter["slot_length"])
    for t in range(horizon):
        time = t*parameter["slot_length"]+parameter["t_start"]+parameter["t_hat"]
        print('********************************* ', time, " ******************************************")

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
        # parameter for current optimization
        t_end = int(min(t+parameter['T_c']/parameter["slot_length"],J_trans_baseline.shape[1]))
        baseline_metric = {
            "J_trans":J_trans_baseline[:,t:t_end],
            "J_idle":J_idle_baseline[t:t_end],
            "P":P_baseline[:,:,t:t_end]
        }

        if t<tau: # during provisioning stage, execute provision model
            c_parameter = parameter_ori.copy()
            c_parameter["T_c"] = (t_end-t)*parameter_ori["slot_length"]
            c_parameter["tau"] = parameter_ori["slot_length"]
            c_parameter["t_start"] = time*10
            c_parameter["t_hat"] = 0

            t_sim_start = Time.time()
            first_sol, _ = provision_model(
                parameter=c_parameter,
                constant=constant,
                initial_state={
                    "vacant":state_overall['vacant'][:,:,t].copy(),
                    "charging":state_overall['charging'][:,:,t].copy(),
                    "waiting":state_overall['waiting'][:,:,t].copy(),
                    "occupied":state_overall['occupied'][:,:,:,t].copy()
                },
                J_trans=J_trans_baseline[:,t:t_end],
                J_idle=J_idle_baseline[t:t_end],
                P=P_baseline[:,:,t:t_end],
                date=date)
            
            t_sim_end = Time.time()
            runtime[t] = t_sim_end - t_sim_start
            
            solution = first_sol

            decision_new = solution["decision"] # type: ignore
            metric_new = solution["metric"]     # type: ignore
            state_new = solution["state"]       # type: ignore
        else: # after provisioning stage, execute response policy
            c_parameter = {
                "t_start":time,
                "t_hat":0,
                "T_b":max(0,parameter["T_b"]-max(0,(t*parameter["slot_length"]-parameter["tau"]))),
                "tau":max(0,parameter["tau"]-(t*parameter["slot_length"])),
                "T_c":(t_end-t)*parameter["slot_length"],
                "slot_length":parameter["slot_length"],
                "epsilon":parameter["epsilon"],
                "ita":parameter["ita"],
                "charging_constraint":parameter["charging_constraint"]
            }

            # apply optimization
            decision_new, metric_new, state_new = potential_opt(
                parameter=c_parameter,
                constant=constant,
                initial_state={
                    "vacant":state_overall['vacant'][:,:,t].copy(),
                    "charging":state_overall['charging'][:,:,t].copy(),
                    "waiting":state_overall['waiting'][:,:,t].copy(),
                    "occupied":state_overall['occupied'][:,:,:,t].copy()
                },
                baseline_metric=baseline_metric,
            demand_data_path=data_path)       

        if decision_new is not None:
            decision = decision_new
            counter = 0
        else:
            counter += 1
            print("***** Infeasible solution, apply decision from %d time slot ago."%(counter))

        if metric_new is not None:
            if "Delta_P" in metric_new.keys():
                state_overall['deltaP_opt'][t] = metric_new['Delta_P'] # power metric
            else:
                state_overall['deltaP_opt'][t] = 1 # energy metric

            state_overall["J_trans_opt"][:,t] = metric_new["J_trans"][:,0]
            
            metric = metric_new
        else:
            state_overall['deltaP_opt'][t] = -1 # optimization failed
            state_overall["J_trans_opt"][:,t] = 0

        # num_served = metric["J_trans"][:,counter]
        num_served = J_trans_baseline[:,t]          # make passenger service consistent to reference

        decision_overall['xp'][:,:,:,t] = decision['xp'][:,:,:,counter].copy()
        decision_overall['xd'][:,:,:,t] = decision['xd'][:,:,:,counter].copy()
        decision_overall['yp'][:,:,:,t] = decision['yp'][:,:,:,counter].copy()
        decision_overall['yd'][:,:,:,t] = decision['yd'][:,:,:,counter].copy()
        decision_overall['u'][:,:,t] = decision['u'][:,:,counter]
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

        taxi_state.append(taxis.copy())
        #endregion
    
    #endregion

    metric_overall = {
        "num_charging":num_charging,
        "J_trans":J_trans,
        "J_idle":J_idle,
        "runtime":runtime
    }

    return decision_overall, metric_overall, state_overall, taxi_state

if __name__ == '__main__':
    # date = "2015-03-14"   # Case 4
    # date = "2015-01-29"   # Case 10
    # date = "2015-02-16"   # case 5
    # date = "2015-01-21"   # Case 9
    # date = "2015-03-25"   # Case 14
    # date = "2015-03-17"   # Case 13
    date = "2015-02-25"   # Case 12
    
    parameter = {
        "t_start":960, # time of simulation starts (unit: min) 1020, 960
        "t_hat":20, # time of beginning of balancing requests (unit: min)
        "tau":60, # ramping down duration (unit: min)
        "T_b":120, # service duration (unit: min)
        "T_c":260, # length of control horizon (unit: min)
        "slot_length":20, # length of timeslot (should be multiple of 10, unit: min)
        "epsilon":0.0,
        "ita":np.inf,
        "charging_constraint":True,
        "max_iter":10,
        "sol_diff_th":10,
        "hyperParameter":{
            "low_initial_battery":0,
            "high_initial_battery":15,
            "max_occupied_length":10
        }
    }

    constant = {
        "L":15,
        "L2":3,
        "L_threshold":2,
        "num_of_taxis":13000, # 13000
        "num_of_regions":38
    }

    note = "verify algorithm and compare with other solution."
    baseline_folder = "12_4-2023_10_28_18_47_58"
    folder_name  = baseline_folder.split("-")[0]

    
    """decision_baseline, metric_baseline, state_baseline, initial_state_baseline = baseline_policy(
        parameter=parameter.copy(),
        constant=constant.copy(),
        date=date)
    baseline_results = [{
        "parameter":parameter,
        "constant":constant,
        "decision_baseline":decision_baseline,
        "metric_baseline":metric_baseline,
        "state_baseline":state_baseline,
        "initial_state_baseline":initial_state_baseline
    }]
    #"""

    baseline_result_path = ROOT.joinpath("results","final_results","Oracle",baseline_folder)
    # baseline_result_path = ROOT.joinpath("utils","singleLevel_solution","test_result")
    baseline_file_name = "test_baseline"
    with open(str(baseline_result_path.joinpath(baseline_file_name)),'rb') as f:
        baseline_results = pickle.load(f)
        parameter_baseline = baseline_results[0]["parameter"]
        metric_baseline = baseline_results[0]['metric_baseline']
        state_baseline = baseline_results[0]['state_baseline']
        decision_baseline = baseline_results[0]['decision_baseline']
        initial_state_baseline = baseline_results[0]["initial_state_baseline"]
    # """

    now = datetime.now()
    result_folder = folder_name+"-%d_%d_%d_%d_%d_%d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
    result_path = ROOT.joinpath("utils","singleLevel_solution","test_result",result_folder)
    os.mkdir(result_path)

    parameter_baseline_json = json.dumps(
        {"parameter":parameter,
         "constant":constant,
         "simulation time":str(datetime.now()),
         "data date":date,
         "baseline_result_folder":str(baseline_result_path),
         "note":note
        },indent=4)
    with open(result_path.joinpath('parameter_constant.json'),'w') as f:
        f.write(parameter_baseline_json)


    t_hat = min(144,int(np.floor(parameter["t_hat"]/parameter["slot_length"])))
    occupied = occupied_taxi_estimate(
        parameter=parameter.copy(),
        supply=state_baseline["supply"],
        t_hat = t_hat,
        initial_occupied_taxis=initial_state_baseline["occupied"],
        constant=constant,
        date=date
    )

    initial_state = {
        "vacant":state_baseline["vacant"][:,:,t_hat],
        "charging":state_baseline["charging"][:,:,t_hat],
        "waiting":state_baseline["waiting"][:,:,t_hat],
        "occupied":occupied
    }

    parameter_pro = parameter.copy()
    parameter_pro["t_start"] += parameter["t_hat"]
    parameter_pro["T_c"] -= parameter["t_hat"]
    parameter_pro["t_hat"] = 0

    T_c = min(144,int(np.floor(parameter_pro["T_c"]/parameter_pro["slot_length"])))
    end_idx = min(T_c+t_hat,metric_baseline["J_trans"].shape[1])

    print("--------Start Provisioning Calculation-----------")
    first_solution, final_solution = provision_model(
        parameter=parameter_pro,
        constant=constant,
        J_trans=metric_baseline["J_trans"][:,t_hat:end_idx],
        J_idle=metric_baseline["J_idle"][t_hat:end_idx],
        P=metric_baseline["num_charging"][:,:,t_hat:end_idx],
        initial_state=initial_state,
        date=date
    )

    with open(result_path.joinpath("first_sol.pickle"),"wb") as f:
        pickle.dump(first_solution,f)

    with open(result_path.joinpath("final_sol.pickle"),"wb") as f:
        pickle.dump(final_solution,f)
    pass
