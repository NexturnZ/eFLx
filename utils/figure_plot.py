import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] # set root directory
sys.path.append(os.path.abspath(ROOT))

""" Function: extract statistics from experiment results
"""
def load_results():
    """ Function: extract charging during the provision stage
    """
    def read_prov_energy(model, folder):
        with open(folder.joinpath("result.pickle"),"rb") as f:
            rst = pickle.load(f)

        with open(folder.joinpath("parameter_constant.json"),"r") as f:
            setting = json.load(f)
            parameter = setting["parameter"]
            constant = setting["constant"]

        battery_capacity = 40e-3 # battery capacity of an e-taxi; unit: MWh
        charging_enengy = constant["L2"]*(battery_capacity/constant["L"]) # energy of charging for one time slot; unit: MWh

        if model != "TRC":
            t_begin_p = 0
        else:
            t_begin_p = int(parameter["t_hat"]/parameter["slot_length"])
        t_end_p = t_begin_p + int(parameter["tau"]/parameter["slot_length"])
        charging = np.sum(rst["decision"]["u"][:,:,t_begin_p:t_end_p]) *charging_enengy
        # charging = np.sum(rst["decision"]["u"][:,:,t_begin_p:t_end_p])

        return charging
    
    result_root_path = ROOT.joinpath("results")
    file_name = "parameter_constant.json"
    # models = ["TRC","TES","R2I","eFlx","Oracle"]
    with open('config.json') as f:
        config = json.load(f)
        models = config["baseline"]

    trials = []

    for m in models:
        dirs = os.listdir(result_root_path.joinpath(m))
        for d in dirs:
            trial_path = result_root_path.joinpath(m,d)
            with open(trial_path.joinpath(file_name),"r") as f:
                trial = json.load(f)
            trial_info = {
                "model":m,
                "tau":trial["parameter"]["tau"],
                "charging_constraint":trial["parameter"]["charging_constraint"],
                "e_ref":trial["flexibility_estimation"]["e_ref"],
                "e_res":trial["flexibility_estimation"]["e_res"],
                "e_res_prov":read_prov_energy(model=m,folder=trial_path),
                "flexibility":trial["flexibility_estimation"]["flexibility"],
                "idle_driving_provision":trial["cost_estimation"]["idle_driving_provision"],
                "idle_wating_provision":trial["cost_estimation"]["idle_wating_provision"],
                "runtime":trial["runtime"]
            }
            trials.append(trial_info)
    
    trials_df = pd.DataFrame(trials)

    battery_capacity = 40e-3 # battery capacity of an e-taxi; unit: MWh
    charging_enengy = 3/15*battery_capacity # energy of charging for one time slot; unit: MWh
    trials_df[["e_res","e_ref","flexibility"]] *= charging_enengy


    trials_df["idle_driving_provision"] /= trial["constant"]["num_of_taxis"]
    trials_df["idle_wating_provision"] *= trial["parameter"]["slot_length"]/trial["constant"]["num_of_taxis"]

    return trials_df

def plot_flexibility():
    font = {
        "family":"Times New Roman",
        "size":"28"
        }
    matplotlib.rc("font",**font)

    with open('config.json') as f:
        config = json.load(f)
        models = config["baseline"]
    
    results_df = load_results() # load data

    fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(16,6))
    ## with charging constraint
    model_order = ["TRC","R2I","eFlx","Oracle"]

    results_wcl = results_df.query("(model in ['R2I','eFlx','Oracle'] and charging_constraint==True and tau==60) or model=='TRC'")
    sns.barplot(data=results_wcl,x="model",y="flexibility",
        order=[m for m in model_order if m in models],color="grey",ax=ax1,capsize=0.15,errwidth=1,errorbar=("sd"))
    ax1.set_ylabel("Flexibility (MWh)",fontsize=28,fontname="Times New Roman")
    ax1.set_xlabel("",fontsize=28,fontname="Times New Roman")
    ax1.set_ylim(bottom=80)
    ax1.set_title("With charging constraint")
        
    ## without charging constraint
    model_order = ["TRC","R2I","TES","eFlx"]
    results_wocl = results_df.query("(model in ['R2I','TES','eFlx'] and charging_constraint==False and tau==60) or model=='TRC'")
    sns.barplot(data=results_wocl,x="model",y="flexibility",
        order=[m for m in model_order if m in models],color="grey",ax=ax2,capsize=0.15,errwidth=1,errorbar=("sd"))
    ax2.set_ylabel("Flexibility (MWh)",fontsize=28,fontname="Times New Roman")
    ax2.set_xlabel("",fontsize=28,fontname="Times New Roman")
    ax2.set_ylim(bottom=80)
    ax2.set_title("Without charging constraint")

    fig.suptitle("Flexibility")

    plt.tight_layout()
    plt.savefig(ROOT.joinpath("figures","flexibility.pdf"),bbox_inches='tight')

""" Function: plot the charging demand during the service stage.
"""
def plot_charging_serv():
    font = {
        "family":"Times New Roman",
        "size":"28"
        }
    matplotlib.rc("font",**font)

    with open('config.json') as f:
        config = json.load(f)
        models = config["baseline"]
    
    results_df = load_results() # load data

    fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(16,6))
    ## with charging constraint
    model_order = ["TRC","R2I","eFlx","Oracle"]
    results_wcl = results_df.query("(model in ['R2I','eFlx','Oracle'] and charging_constraint==True and tau==60) or model=='TRC'")
    sns.barplot(data=results_wcl,x="model",y="e_res",
        order=[m for m in model_order if m in models],color="grey",ax=ax1,capsize=0.15,errwidth=1,errorbar=("sd"))
    ax1.set_ylabel("Charging demand (MWh)",fontsize=28,fontname="Times New Roman")
    ax1.set_xlabel("",fontsize=28,fontname="Times New Roman")
    ax1.set_title("With charging constraint")
    ax1.set_ylim(bottom=0)
        
    ## without charging constraint
    model_order = ["TRC","R2I","TES","eFlx"]
    results_wocl = results_df.query("(model in ['R2I','TES','eFlx'] and charging_constraint==False and tau==60) or model=='TRC'")
    sns.barplot(data=results_wocl,x="model",y="e_res",
        order=[m for m in model_order if m in models],color="grey",ax=ax2,capsize=0.15,errwidth=1,errorbar=("sd"))
    ax2.set_ylabel("Charging demand (MWh)",fontsize=28,fontname="Times New Roman")
    ax2.set_xlabel("",fontsize=28,fontname="Times New Roman")
    ax2.set_title("Without charging constraint")
    ax2.set_ylim(bottom=0)

    fig.suptitle("Charging demand during the service stage")

    plt.tight_layout()
    plt.savefig(ROOT.joinpath("figures","charging_demand_during_service_stage.pdf"),bbox_inches='tight')

""" Function: plot the charging demand during the provision stage.
"""
def plot_charging_prov():
    font = {
        "family":"Times New Roman",
        "size":"28"
        }
    matplotlib.rc("font",**font)

    with open('config.json') as f:
        config = json.load(f)
        models = config["baseline"]
    
    results_df = load_results() # load data

    fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(16,6))
    ## with charging constraint
    model_order = ["TRC","R2I","eFlx","Oracle"]
    results_wcl = results_df.query("(model in ['R2I','eFlx','Oracle'] and charging_constraint==True and tau==60) or model=='TRC'")
    sns.barplot(data=results_wcl,x="model",y="e_res_prov",
        order=[m for m in model_order if m in models],color="grey",ax=ax1,capsize=0.15,errwidth=1,errorbar=("sd"))
    ax1.set_ylabel("Charging demand (MWh)",fontsize=28,fontname="Times New Roman")
    ax1.set_xlabel("",fontsize=28,fontname="Times New Roman")
    ax1.set_title("With charging constraint")
    ax1.set_ylim(bottom=0)
        
    ## without charging constraint
    model_order = ["TRC","R2I","TES","eFlx"]
    results_wocl = results_df.query("(model in ['R2I','TES','eFlx'] and charging_constraint==False and tau==60) or model=='TRC'")
    sns.barplot(data=results_wocl,x="model",y="e_res_prov",
        order=[m for m in model_order if m in models],color="grey",ax=ax2,capsize=0.15,errwidth=1,errorbar=("sd"))
    ax2.set_ylabel("Charging demand (MWh)",fontsize=28,fontname="Times New Roman")
    ax2.set_xlabel("",fontsize=28,fontname="Times New Roman")
    ax2.set_title("Without charging constraint")
    ax2.set_ylim(bottom=0)

    fig.suptitle("Charging demand during the provision stage")

    plt.tight_layout()
    plt.savefig(ROOT.joinpath("figures","charging_demand_during_provision_stage.pdf"),bbox_inches='tight')

""" Function: plot the idle driving distance and idle waiting time.
"""
def plot_idle():
    font = {
        "family":"Times New Roman",
        "size":"28"
    }
    matplotlib.rc("font",**font)

    with open('config.json') as f:
        config = json.load(f)
        models = config["baseline"]
    
    results_df = load_results() # load data
    results_df = results_df.query("(model in ['R2I','eFlx','Oracle'] and charging_constraint==True and tau==60) or model in ['TRC','TES']")

    fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(16,7))
    ## average idle driving distance
    model_order = ["TES","TRC","R2I","eFlx","Oracle"]
    sns.barplot(data=results_df,x="model",y="idle_driving_provision",
        order=[m for m in model_order if m in models],color="grey",ax=ax1,capsize=0.15,errwidth=1,errorbar=("sd"))
    ax1.set_ylabel("Average idle distance (mile)",fontsize=28,fontname="Times New Roman")
    ax1.set_xlabel("",fontsize=28,fontname="Times New Roman")
    ax1.set_ylim(bottom=7)
        
    ## average idle waiting time
    model_order = ["TES","TRC","R2I","eFlx","Oracle"]
    sns.barplot(data=results_df,x="model",y="idle_wating_provision",
        order=[m for m in model_order if m in models],color="grey",ax=ax2,capsize=0.15,errwidth=1,errorbar=("sd"))
    ax2.set_ylabel("Average idle waiting time (min)",fontsize=28,fontname="Times New Roman")
    ax2.set_xlabel("",fontsize=28,fontname="Times New Roman")
    ax2.set_ylim(bottom=0)

    fig.suptitle("Overhead during provision stage.")

    plt.tight_layout()
    plt.savefig(ROOT.joinpath("figures","idle.pdf"),bbox_inches='tight')

""" Function: plot showcase for a single trial.
"""
def trial_showcase():

    trial_no = "2"

    result_folders = {
        "TRC":"TRC/%s"%(trial_no),
        "RES":"Oracle/%s_1"%(trial_no),
        "PRO":"eFlx/%s_1"%(trial_no),
        "R2I":"R2I/%s_1"%(trial_no)
    }

    root_folder = ROOT.joinpath("results")

    horizon = 3

    trial_results = {}

    # load ref
    dir = root_folder.joinpath(result_folders["TRC"])
    with open(dir.joinpath("parameter_constant.json"),"r") as f:
        setting = json.load(f)
        parameter = setting["parameter"]
        constant = setting["constant"]
        t_hat = int(parameter["t_hat"]/parameter["slot_length"])

    battery_capacity = 40e-3 # battery capacity of an e-taxi; unit: MWh
    charging_power = constant["L2"]*(battery_capacity/constant["L"])/(parameter["slot_length"]/60)

    with open(dir.joinpath("result.pickle"), "rb") as f:
        mpc_result = pickle.load(f)
        charging_mpc = np.sum(mpc_result["state"]["charging"][:,:,t_hat:],(0,1))*charging_power

    trial_results["REF"] = charging_mpc

    # load baselines
    for k in result_folders:
        dir = root_folder.joinpath(result_folders[k])

        if k == "TRC":
            t_hat = 1
        else:
            t_hat = 0

        with open(dir.joinpath("result.pickle"), "rb") as f:
            mpc_result = pickle.load(f)
            charging_mpc = np.sum(mpc_result["state"]["charging"][:,:,t_hat:t_hat+horizon],(0,1))*charging_power

        with open(dir.joinpath("flexibility_evaluation.pickle"),"rb") as f:
            flx_result = pickle.load(f)
            charging_flx = np.sum(flx_result["res_simulation"]["state"]["charging"],(0,1))*charging_power

        charging = np.concatenate((charging_mpc,charging_flx))

        trial_results[k] = charging


    trial_results["REF"] = trial_results["REF"][:len(trial_results["PRO"])]

    num_ts = len(trial_results["PRO"])
    start = parameter["t_start"] + parameter["t_hat"]
    step = parameter["slot_length"]
    end = start + step * (num_ts - 1)
    time = np.linspace(start, end, num_ts)/60

    font = {
        "family":"Times New Roman",
        "size":"32"
    }
    matplotlib.rc("font",**font)

    fig, ax = plt.subplots(figsize=(10,8))
    # ax2 = plt.twinx(ax=ax1)

    markers = {
        "TRC":".",
        "RES":"^",
        "PRO":"D",
        "TES":"+",
        "R2I":"s",
        "REF":"o"
    }

    name = {
        "TRC":"TRC","RES":"Oracle","PRO":"eFlx","TES":"TES","R2I":"R2I","REF":"Reference policy"
    }

    sols = ["REF","TRC","R2I","PRO","RES"]
    for k in trial_results:
        if k in sols:
            ax.plot(time[:-1],trial_results[k][:-1],label=name[k],marker=markers[k],markersize=14,linewidth=3)

    ax.grid(True)
    ax.set_xlabel("Time of day (h)")
    ax.set_ylabel("Charging demand (MW)")
    ax.legend(loc=(-0.05,1.01),ncol=3,fontsize=24)

    # label provision stage and service stage
    plt.axvspan(17.33, 19.33, color='yellow', alpha=0.3)
    plt.axvline(x=17.33, color='red', linestyle='--')
    plt.axvline(x=19.33, color='red', linestyle='--')

    plt.text(16.3, 40, 'Provision\n   Stage', fontsize=32, color='black')
    plt.text(18.0, 40, 'Service\n Stage', fontsize=32, color='black')

    plt.tight_layout()
    plt.savefig(ROOT.joinpath("figures","trial_showcase.pdf"),bbox_inches='tight')

def plot_result_figures():
    os.makedirs(ROOT.joinpath("figures"),exist_ok=True)

    plot_flexibility()
    plot_charging_serv()
    plot_charging_prov()
    plot_idle()
    trial_showcase()

if __name__ == '__main__':
    plot_result_figures()
    # print(load_results())
