import numpy as np
from matplotlib import pyplot as plt
import pickle
from datetime import datetime
import json
import pandas as pd
import time

import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[0] # set root directory
sys.path.append(os.path.abspath(ROOT))

from utils.mpc_trial_recovery import idle_recovery
from utils.figure_plot import plot_result_figures
from utils.flexibility_prediction.baseline_policy import baseline_policy_mpc
from utils.flexibility_prediction.potential_policy import potential_policy_mpc, potential_policy
from utils.singleLevel_solution.singleLevel_solution import provision_model_mpc
from utils.baseline_strategies.R2E import TES_policy_mpc
from utils.baseline_strategies.R2I import R2I_policy_mpc

""" Function: evaluate flexibility of a given taxi state
    Input:
        1. parameter
        2. constant
        3. taxis: e-taxi fleet state of which flexibility is estimated;
        4. date: date of simulation
        5. J_trans_ref: external reference of passenger service
"""
def flexibility_evaluation_selfRef(model,result_path:Path):
    T_c_eval = 180
    eval_result_file_name = "flexibility_evaluation.pickle"
    
    #region: load result
    with open(result_path.joinpath("parameter_constant.json"),"r") as f:
        setting = json.load(f)
        parameter = setting["parameter"]
        constant = setting["constant"]
        tau = int(parameter["tau"]/parameter["slot_length"])
        t_hat = int(parameter["t_hat"]/parameter["slot_length"])
        T_b = int(parameter["T_b"]/parameter["slot_length"])
        max_occupied_length = parameter["hyperParameter"]["high_initial_battery"]  # maximum trip length of occupied taxis, unit: 10 min

        date = setting["data date"]
    
    with open(result_path.joinpath("result.pickle"),"rb") as f:
        result = pickle.load(f)

    # load TRC for J_trans
    c = result_path.name.split("_")[0]
    root_folder = result_path.parents[1]
    TRC_folders = os.listdir(root_folder.joinpath("TRC"))
    for dir in TRC_folders:
        if dir.split("_")[0] == c:
            break
    
    with open(root_folder.joinpath("TRC",dir,"result.pickle"),"rb") as f:
        ref_results = pickle.load(f)
        J_trans_ref = ref_results["metric"]["J_trans"]
        J_idle_ref = ref_results["metric"]["J_idle"]
        P_ref = ref_results["metric"]["num_charging"]

    #endregion
    
    # load reference charging
    if model == "TRC":
        t_begin = tau+t_hat
        t_begin_ref = tau+t_hat
    else:
        t_begin = tau
        t_begin_ref = tau+t_hat
    t_end = t_begin_ref + int(T_c_eval/parameter["slot_length"])

    if model in ["Oracle","R2I"]:
        e_ref = np.sum(ref_results["decision"]["u"][:,:,t_begin_ref:t_begin_ref+T_b])
    elif "demand_prediction_error" in parameter.keys():
        if parameter["demand_prediction_error"] > 0:
            # e_ref = np.sum(ref_results["decision"]["u"][:,:,t_begin_ref:t_begin_ref+T_b])
            e_ref = np.sum(result["decision"]["u"][:,:,t_begin:t_begin+T_b])
        else:
            e_ref = np.sum(result["decision"]["u"][:,:,t_begin:t_begin+T_b])
    else:
        e_ref = np.sum(result["decision"]["u"][:,:,t_begin:t_begin+T_b])

    #region: simulate response charging
    # initial_state = {
    #     "vacant":result["state"]["vacant"][:,:,t_begin],
    #     "charging":result["state"]["charging"][:,:,t_begin],
    #     "waiting":result["state"]["waiting"][:,:,t_begin],
    #     "occupied":result["state"]["occupied"][:,:,:,t_begin],
    # }

    initial_state = {
        "vacant":np.zeros((constant['num_of_regions'],constant['L'])),
        "charging":np.zeros((constant['num_of_regions'],constant['L'])),
        "waiting":np.zeros((constant['num_of_regions'],constant['L'])),
        "occupied":np.zeros((constant['num_of_regions'],constant['L'],max_occupied_length)),
    }

    # calculate fleet state with status of e-taxis
    taxis = result["taxi_state"][t_begin]
    ratio = 0.5
    idle_dis_dt = 2.5 * parameter["slot_length"]/10
    for n in range(constant['num_of_taxis']):
        l = min(int(taxis.loc[n,"remaining_energy"]/(idle_dis_dt/ratio)),constant['L']-1) # calcuate energy level
        if taxis.loc[n,"status"] == 3: # the taxi is vacant
            i = taxis.loc[n,"location"]
            initial_state['vacant'][i,l] += 1 # type: ignore
        elif taxis.loc[n,"status"] == 1: # the taxi is waiting
            i = taxis.loc[n,"location"]
            initial_state['waiting'][i,l] += 1 # type: ignore
        elif taxis.loc[n,"status"] == 0: # the taxi is charging
            i = taxis.loc[n,"location"]
            initial_state['charging'][i,l] += 1 # type: ignore
        elif taxis.loc[n,"status"] == 2: # the taxi is occupied
            i = taxis.loc[n,"destination"]
            arriving_time = int((taxis.loc[n,"arriving_time"]-parameter["t_start"])/parameter["slot_length"]-t_begin_ref) # type: ignore
            if arriving_time == 0:
                initial_state["vacant"][i,l] += 1 # type: ignore
            elif arriving_time < max_occupied_length:
                initial_state['occupied'][i,l,arriving_time] += 1 # type: ignore
    

    parameter_eval = parameter.copy()
    parameter_eval["t_start"] = parameter["t_start"]+parameter["t_hat"]+parameter["tau"]
    parameter_eval["t_hat"] = 0
    parameter_eval["tau"] = 0
    parameter_eval["T_c"] = T_c_eval
    if "demand_prediction_error" in parameter.keys():
        parameter_eval["demand_prediction_error"]=0
    
    decision_res, metric_res, state_res = potential_policy(
            parameter=parameter_eval,
            constant=constant,
            J_trans=J_trans_ref[:,t_begin_ref:t_end], # take the value from TRC
            J_idle=J_idle_ref[t_begin_ref:t_end],
            P=P_ref[:,:,t_begin_ref:t_end],
            initial_state=initial_state,
            date=date
        )
    e_res = np.sum(decision_res["u"][:,:,:T_b])  # type: ignore
    #endregion

    # calculate flexibility & story result
    flexibility_estimation = {
        "T_c_eval":T_c_eval,
        "flexibility":e_ref-e_res,
        "e_ref":e_ref,
        "e_res":e_res
    }

    setting["flexibility_estimation"] = flexibility_estimation
    setting_str = json.dumps(setting,indent=4)
    with open(result_path.joinpath("parameter_constant.json"),"w") as f:
        f.write(setting_str)

    # save response simulation
    eval_result = {
        "flexibility_estimation":flexibility_estimation,
        "res_simulation":{
            "decision":decision_res,
            "state":state_res,
            "metric":metric_res
        }
    }
    with open(result_path.joinpath(eval_result_file_name),"wb") as f:
        pickle.dump(eval_result,f)

    pass

""" Function: conduct MPC simulation for TRC
"""
def main_reference_policy(date,folder_name,parameter):
    start_time = time.time()
    parameter = parameter.copy()
    result_file_name = "result.pickle"
    estimate_flexibility = True
    
    constant = {
        "L":15, # number of energy level (=30 when slot_lengh = 10)
        "L2":3, # (=6 when slot_lengh = 10)
        "L_threshold":2, # 4 when slot_lengh = 10
        "num_of_taxis":13000, # 13000
        "num_of_regions":38
    }

    note = "Simulation for TRC"

    root_folder_name = "TRC"
    now = datetime.now()
    # result_folder = folder_name+"-%d_%d_%d_%d_%d_%d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
    result_folder = folder_name
    result_path = ROOT.joinpath("results/%s/%s"%(root_folder_name,result_folder))
    os.makedirs(result_path,exist_ok=True)
    if "flexibility_evaluation.pickle" in os.listdir(result_path):
        return
    else:
        files = os.listdir(result_path)
        for file in files:
            os.remove(result_path.joinpath(file))



    setting = {
        "Model":"Reference model",
        "parameter":parameter,
        "constant":constant,
        "simulation time":str(now),
        "data date":date,
        "note":note
    }
    parameter_baseline_json = json.dumps(setting,indent=4)
    with open(result_path.joinpath('parameter_constant.json'),'w') as f:
        f.write(parameter_baseline_json)

    print("--------------------Baseline Policy---------------------------------")
    #region: start MPC simulation
    init_state_file = ROOT.joinpath("data","initial_state",folder_name+".pickle")
    with open(init_state_file,"rb") as f:
        initial_state = pickle.load(f)
        parameter["T_c"] = 160

    decision_baseline, metric_baseline, state_baseline, taxi_state_baseline = baseline_policy_mpc(
        parameter=parameter.copy(),
        constant=constant.copy(),
        initial_state=initial_state,
        date=date)
    baseline_result = {
        "parameter":parameter,
        "constant":constant,
        "decision":decision_baseline,
        "metric":metric_baseline,
        "state":state_baseline,
        "taxi_state":taxi_state_baseline
    }
    
    # write baseline result into file
    with open(str(result_path.joinpath(result_file_name)),'wb') as f:
        pickle.dump(baseline_result,f)
    print("Experiment path:",str(result_path))
    #endregion

    #region: evaluate flexibility after provisioning
    if estimate_flexibility:
        model = "TRC"
        flexibility_evaluation_selfRef(model=model,result_path=result_path)
    #endregion

    # retrieve idle driving distance and idle waiting time
    idle_recovery(model=model,result_folder=result_path,save=True)

    # record runtime
    end_time = time.time()
    with open(result_path.joinpath('parameter_constant.json'),'r') as f:
        parameter_constant = json.load(f)

    parameter_constant["runtime"] = end_time-start_time

    parameter_constant_str = json.dumps(parameter_constant,indent=4)
    with open(result_path.joinpath('parameter_constant.json'),'w') as f:
        f.write(parameter_constant_str)

""" Function: conduct MPC simulation for Oracle
"""
def main_response_policy(folder_name:str,parameter):
    start_time = time.time()
    parameter = parameter.copy()
    result_root_folder = "results"
    baseline_result_path = ROOT.joinpath(result_root_folder,"TRC",folder_name.split("_")[0])
    result_file_name = "result.pickle"

    estimate_flexibility = True
     
    constant = {
        "L":15, # number of energy level (=30 when slot_lengh = 10)
        "L2":3, # (=6 when slot_lengh = 10)
        "L_threshold":2, # 4 when slot_lengh = 10
        "num_of_taxis":13000, # 13000
        "num_of_regions":38
    }

    note = "Simulation for Oracle"

    with open(baseline_result_path.joinpath("parameter_constant.json"),"r") as f:
        tmp = json.load(f)
        date = tmp["data date"]

    root_folder_name = "Oracle"
    now = datetime.now()
    result_folder = folder_name
    result_path = ROOT.joinpath(result_root_folder,"%s/%s"%(root_folder_name,result_folder))
    os.makedirs(result_path,exist_ok=True)
    if "flexibility_evaluation.pickle" in os.listdir(result_path):
        return
    else:
        files = os.listdir(result_path)
        for file in files:
            os.remove(result_path.joinpath(file))

    setting =  {
            "Model":"Response model",
            "parameter":parameter,
            "constant":constant,
            "simulation time":str(now),
            "data date":date,
            "ref_policy_path":str(baseline_result_path),
            "note":note
        }
    setting_str = json.dumps(setting,indent=4)
    with open(result_path.joinpath('parameter_constant.json'),'w') as f:
        f.write(setting_str)

    #region: load reference policy
    t_hat = min(144,int(np.floor(parameter["t_hat"]/parameter["slot_length"])))

    with open(str(baseline_result_path.joinpath("result.pickle")),'rb') as f:
        baseline_results = pickle.load(f)
        metric_baseline = baseline_results['metric']
        initial_state = baseline_results["taxi_state"][t_hat]
    #endregion
    
    #region: start MPC simulation
    decision_potential, metric_potential, state_potential, taxi_state_potential = potential_policy_mpc(
        parameter=parameter.copy(),
        constant=constant.copy(),
        J_trans_baseline=metric_baseline['J_trans'][:,t_hat:],
        J_idle_baseline=metric_baseline['J_idle'][t_hat:],
        initial_state=initial_state,
        P_baseline=metric_baseline['num_charging'][:,:,t_hat:],
        date=date
    )

    potential_result = {
        "parameter":parameter,
        "constant":constant,
        "decision":decision_potential,
        "metric":metric_potential,
        "state":state_potential,
        "taxi_state":taxi_state_potential
    }
    with open(str(result_path.joinpath(result_file_name)),'wb') as f:
        pickle.dump(potential_result,f)
    print("Experiment path:",str(result_path))
    #endregion

    #region: evaluate flexibility after provisioning
    if estimate_flexibility:
        model = "Oracle"
        flexibility_evaluation_selfRef(model=model,result_path=result_path)
    #endregion

    # retrieve idle driving distance and idle waiting time
    idle_recovery(model=model,result_folder=result_path,save=True)

    # record runtime
    end_time = time.time()
    with open(result_path.joinpath('parameter_constant.json'),'r') as f:
        parameter_constant = json.load(f)

    parameter_constant["runtime"] = end_time-start_time

    parameter_constant_str = json.dumps(parameter_constant,indent=4)
    with open(result_path.joinpath('parameter_constant.json'),'w') as f:
        f.write(parameter_constant_str)

""" Function: conduct MPC simulation for provision model
"""
def main_provision_policy(folder_name:str,parameter):
    start_time = time.time()
    parameter = parameter.copy()
    result_root_folder = "results"
    baseline_result_path = ROOT.joinpath(result_root_folder,"TRC",folder_name.split("_")[0])
    result_file_name = "result.pickle"

    estimate_flexibility = True
    
    constant = {
        "L":15, # number of energy level (=30 when slot_lengh = 10)
        "L2":3, # (=6 when slot_lengh = 10)
        "L_threshold":2, # 4 when slot_lengh = 10
        "num_of_taxis":13000, # 13000
        "num_of_regions":38
    }

    note = "Simulation for eFlx"

    with open(baseline_result_path.joinpath("parameter_constant.json"),"r") as f:
        tmp = json.load(f)
        date = tmp["data date"]

    root_folder_name = "eFlx"
    now = datetime.now()
    # result_folder = folder_name+"-%d_%d_%d_%d_%d_%d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
    result_folder = folder_name
    result_path = ROOT.joinpath(result_root_folder,"%s/%s"%(root_folder_name,result_folder))
    os.makedirs(result_path,exist_ok=True)
    if "flexibility_evaluation.pickle" in os.listdir(result_path):
        return
    else:
        files = os.listdir(result_path)
        for file in files:
            os.remove(result_path.joinpath(file))

    setting = {
        "Model":"provisioning model",
        "parameter":parameter,
        "constant":constant,
        "simulation time":str(now),
        "data date":date,
        "ref_policy_path":str(baseline_result_path),
        "note":note
    }

    setting_str = json.dumps(setting,indent=4)
    with open(result_path.joinpath('parameter_constant.json'),'w') as f:
        f.write(setting_str)

    # non-responding baseline
    parameter = parameter.copy()
    parameter["tau"] = parameter["horizon"]

    #region: load reference policy
    t_hat = min(144,int(np.floor(parameter["t_hat"]/parameter["slot_length"])))

    with open(str(baseline_result_path.joinpath("result.pickle")),'rb') as f:
        baseline_results = pickle.load(f)
        metric_baseline = baseline_results['metric']
        initial_state = baseline_results["taxi_state"][t_hat]
    #endregion

    #region: start MPC simulation
    decision, metric, state, taxi_state = provision_model_mpc(
        parameter=parameter.copy(),
        constant=constant.copy(),
        J_trans_baseline=metric_baseline['J_trans'][:,t_hat:],
        J_idle_baseline=metric_baseline['J_idle'][t_hat:],
        initial_state=initial_state,
        P_baseline=metric_baseline['num_charging'][:,:,t_hat:],
        date=date
    )

    potential_result = {
        "parameter":parameter,
        "constant":constant,
        "decision":decision,
        "metric":metric,
        "state":state,
        "taxi_state":taxi_state
    }
    with open(str(result_path.joinpath(result_file_name)),'wb') as f:
        pickle.dump(potential_result,f)
    print("Experiment path:",str(result_path))
    #endregion

    #region: evaluate flexibility after provisioning
    if estimate_flexibility:
        model = "eFlx"
        flexibility_evaluation_selfRef(model=model,result_path=result_path)
    #endregion

    # retrieve idle driving distance and idle waiting time
    idle_recovery(model=model,result_folder=result_path,save=True)

    # record runtime
    end_time = time.time()
    with open(result_path.joinpath('parameter_constant.json'),'r') as f:
        parameter_constant = json.load(f)

    parameter_constant["runtime"] = end_time-start_time

    parameter_constant_str = json.dumps(parameter_constant,indent=4)
    with open(result_path.joinpath('parameter_constant.json'),'w') as f:
        f.write(parameter_constant_str)

""" Function: conduct MPC simulation for TES
"""
def main_TES_policy(folder_name:str,parameter):
    start_time = time.time()
    result_root_folder = "results"
    baseline_result_path = ROOT.joinpath(result_root_folder,"TRC",folder_name.split("_")[0])
    result_file_name = "result.pickle"

    estimate_flexibility = True
    
    constant = {
        "L":15, # number of energy level (=30 when slot_lengh = 10)
        "L2":3, # (=6 when slot_lengh = 10)
        "L_threshold":2, # 4 when slot_lengh = 10
        "num_of_taxis":13000, # 13000
        "num_of_regions":38
    }

    note = "Simulation for TES"

    with open(baseline_result_path.joinpath("parameter_constant.json"),"r") as f:
        tmp = json.load(f)
        date = tmp["data date"]

    root_folder_name = "TES"
    now = datetime.now()
    result_folder = folder_name
    result_path = ROOT.joinpath(result_root_folder,"%s/%s"%(root_folder_name,result_folder))
    os.makedirs(result_path,exist_ok=True)
    if "flexibility_evaluation.pickle" in os.listdir(result_path):
        return
    else:
        files = os.listdir(result_path)
        for file in files:
            os.remove(result_path.joinpath(file))

    setting = {
        "Model":"TES model",
        "parameter":parameter,
        "constant":constant,
        "simulation time":str(now),
        "data date":date,
        "ref_policy_path":str(baseline_result_path),
        "note":note
    }

    setting_str = json.dumps(setting,indent=4)
    with open(result_path.joinpath('parameter_constant.json'),'w') as f:
        f.write(setting_str)

    # non-responding baseline
    parameter = parameter.copy()
    parameter["tau"] = parameter["horizon"] 

    #region: load reference policy
    t_hat = min(144,int(np.floor(parameter["t_hat"]/parameter["slot_length"])))

    with open(str(baseline_result_path.joinpath("result.pickle")),'rb') as f:
        baseline_results = pickle.load(f)
        metric_baseline = baseline_results['metric']
        initial_state = baseline_results["taxi_state"][t_hat]
        parameter["T_c"] = 140
    #endregion

    #region: start MPC simulation
    decision, metric, state, taxi_state = TES_policy_mpc(
        parameter=parameter.copy(),
        constant=constant.copy(),
        J_trans_baseline=metric_baseline['J_trans'][:,t_hat:],
        J_idle_baseline=metric_baseline['J_idle'][t_hat:],
        initial_state=initial_state,
        P_baseline=metric_baseline['num_charging'][:,:,t_hat:],
        date=date
    )

    potential_result = {
        "parameter":parameter,
        "constant":constant,
        "decision":decision,
        "metric":metric,
        "state":state,
        "taxi_state":taxi_state
    }
    with open(str(result_path.joinpath(result_file_name)),'wb') as f:
        pickle.dump(potential_result,f)
    print("Experiment path:",str(result_path))
    #endregion

    #region: evaluate flexibility after provisioning
    if estimate_flexibility:
        model = "TES"
        flexibility_evaluation_selfRef(model=model,result_path=result_path)
    #endregion

    # retrieve idle driving distance and idle waiting time
    idle_recovery(model=model,result_folder=result_path,save=True)

    # record runtime
    end_time = time.time()
    with open(result_path.joinpath('parameter_constant.json'),'r') as f:
        parameter_constant = json.load(f)

    parameter_constant["runtime"] = end_time-start_time

    parameter_constant_str = json.dumps(parameter_constant,indent=4)
    with open(result_path.joinpath('parameter_constant.json'),'w') as f:
        f.write(parameter_constant_str)

""" Function: conduct MPC simulation for R2I
"""
def main_R2I_policy(folder_name:str,parameter):
    start_time = time.time()
    parameter = parameter.copy()
    result_root_folder = "results"
    baseline_result_path = ROOT.joinpath(result_root_folder,"TRC",folder_name.split("_")[0])
    result_file_name = "result.pickle"

    estimate_flexibility = True
     
    constant = {
        "L":15, # number of energy level (=30 when slot_lengh = 10)
        "L2":3, # (=6 when slot_lengh = 10)
        "L_threshold":2, # 4 when slot_lengh = 10
        "num_of_taxis":13000, # 13000
        "num_of_regions":38
    }

    note = "Simulation for R2I"

    with open(baseline_result_path.joinpath("parameter_constant.json"),"r") as f:
        tmp = json.load(f)
        date = tmp["data date"]

    root_folder_name = "R2I"
    now = datetime.now()
    # result_folder = folder_name+"-%d_%d_%d_%d_%d_%d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
    result_folder = folder_name
    result_path = ROOT.joinpath(result_root_folder,"%s/%s"%(root_folder_name,result_folder))
    os.makedirs(result_path,exist_ok=True)
    if "flexibility_evaluation.pickle" in os.listdir(result_path):
        return
    else:
        files = os.listdir(result_path)
        for file in files:
            os.remove(result_path.joinpath(file))

    setting =  {
            "Model":"R2I model",
            "parameter":parameter,
            "constant":constant,
            "simulation time":str(now),
            "data date":date,
            "ref_policy_path":str(baseline_result_path),
            "note":note
        }
    setting_str = json.dumps(setting,indent=4)
    with open(result_path.joinpath('parameter_constant.json'),'w') as f:
        f.write(setting_str)

    # non-responding baseline
    parameter = parameter.copy()
    parameter["tau"] = parameter["horizon"]

    #region: load reference policy
    t_hat = min(144,int(np.floor(parameter["t_hat"]/parameter["slot_length"])))

    with open(str(baseline_result_path.joinpath("result.pickle")),'rb') as f:
        baseline_results = pickle.load(f)
        metric_baseline = baseline_results['metric']
        initial_state = baseline_results["taxi_state"][t_hat]
    #endregion
    
    #region: start MPC simulation
    decision_potential, metric_potential, state_potential, taxi_state_potential = R2I_policy_mpc(
        parameter=parameter.copy(),
        constant=constant.copy(),
        J_trans_baseline=metric_baseline['J_trans'][:,t_hat:],
        J_idle_baseline=metric_baseline['J_idle'][t_hat:],
        initial_state=initial_state,
        P_baseline=metric_baseline['num_charging'][:,:,t_hat:],
        date=date
    )

    potential_result = {
        "parameter":parameter,
        "constant":constant,
        "decision":decision_potential,
        "metric":metric_potential,
        "state":state_potential,
        "taxi_state":taxi_state_potential
    }
    with open(str(result_path.joinpath(result_file_name)),'wb') as f:
        pickle.dump(potential_result,f)
    print("Experiment path:",str(result_path))  
    #endregion

    #region: evaluate flexibility after provisioning
    if estimate_flexibility:
        model = "R2I"
        flexibility_evaluation_selfRef(model=model,result_path=result_path)
    #endregion

    # retrieve idle driving distance and idle waiting time
    idle_recovery(model=model,result_folder=result_path,save=True)

    # record runtime
    end_time = time.time()
    with open(result_path.joinpath('parameter_constant.json'),'r') as f:
        parameter_constant = json.load(f)

    parameter_constant["runtime"] = end_time-start_time

    parameter_constant_str = json.dumps(parameter_constant,indent=4)
    with open(result_path.joinpath('parameter_constant.json'),'w') as f:
        f.write(parameter_constant_str)

def main():
    with open('config.json') as f:
        config = json.load(f)
        models = config["baseline"]

    os.makedirs(ROOT.joinpath("results"),exist_ok=True)
    for m in models:
        os.makedirs(ROOT.joinpath("results",m),exist_ok=True)

    # load trial dates
    with open(ROOT.joinpath("data","trial_dates.json"),"r") as f:
        trial_dates = json.load(f)["trial_dates"]

    #region: define parameters
    # parameter for trials with charging constraint
    parameter_wcl = {
        "t_start":960, # time of simulation starts (unit: min) 4pm
        "t_hat":20, # time of beginning of balancing requests (unit: min)
        "tau":60, # ramping down duration (unit: min) t_hat+tau should be 70 so that the balancing window will start at 5:10pm
        "T_b":120, # service duration (unit: min)
        "T_c":200, # length of control horizon in mpc(unit: min)
        "horizon":200, # total control horizon
        "slot_length":20,           # length of timeslot (should be multiple of 10, unit: min)
        "epsilon":0.0,
        "ita":np.inf,
        "charging_constraint":True, # whether to have charging constraint regarding to TRC during the provision stage 
        "use_first_sol":True,      # whether to use the inital solution of eFlx
        "max_iter":10,              # maximum number of iterations in provision model
        "sol_diff_th":10,           # threshold of convergence
        # "demand_prediction_error":0.1,
        "hyperParameter":{
            "low_initial_battery":3,
            "high_initial_battery":15,
            "max_occupied_length":10
        } 
    }

    # parameter for trials without charging constraint
    parameter_wocl = parameter_wcl.copy()
    parameter_wocl["charging_constraint"] = False

    # parameter for TRC trials
    parameter_trc = parameter_wcl.copy()
    parameter_trc["horizon"] = 300

    # parameters for eFlx under different provision duration
    parameter_tau40 = parameter_wcl.copy()
    parameter_tau40["tau"] = 40
    parameter_tau40["t_hat"] = 40

    parameter_tau20 = parameter_wcl.copy()
    parameter_tau20["tau"] = 20
    parameter_tau20["t_hat"] = 60
    #endregion

    start_time = time.time()
    for m in models:
        for idx in trial_dates.keys():
            data_date = trial_dates[idx]
            if m=="TRC":    # trials for TRC
                folder_name = idx       # tau=60
                main_reference_policy(date=data_date, folder_name=folder_name,parameter=parameter_trc)

            elif m=="eFlx": # trials for eFlx
                folder_name = idx+"_1"  # tau=60 & with charging constraint
                main_provision_policy(folder_name=folder_name,parameter=parameter_wcl)

                folder_name = idx+"_2"  # tau=60 & without charging constraint
                main_provision_policy(folder_name=folder_name,parameter=parameter_wocl)

                folder_name = idx+"_3"  # tau=40 & with charging constraint
                main_provision_policy(folder_name=folder_name,parameter=parameter_tau40)

                folder_name = idx+"_4"  # tau=20 & with charging constraint
                main_provision_policy(folder_name=folder_name,parameter=parameter_tau20)

            elif m=="Oracle":   # trials for Oracle
                folder_name = idx+"_1"  # tau=60 & with charging constraint
                main_response_policy(folder_name=folder_name,parameter=parameter_wcl)

            elif m=="R2I":      # trials for R2I
                folder_name = idx+"_1"  # tau=60 & with charging constraint
                main_R2I_policy(folder_name=folder_name,parameter=parameter_wcl)

                folder_name = idx+"_2"  # tau=60 & without charging constraint
                main_R2I_policy(folder_name=folder_name,parameter=parameter_wocl)

            elif m=="TES":      # trials for TES
                folder_name = idx+"_2"  # tau=60 & with charging constraint
                main_TES_policy(folder_name=folder_name,parameter=parameter_wocl)

    ## plot result figures
    plot_result_figures()

    end_time = time.time()
    print("runtime (sec): ", end_time-start_time)

if __name__ =='__main__':
    main()

    