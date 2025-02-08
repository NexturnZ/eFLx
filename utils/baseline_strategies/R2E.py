from gurobipy import Model, GRB, GurobiError, LinExpr, GenExpr, abs_
import random
import pickle
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2] # set root directory
sys.path.append(os.path.abspath(ROOT))

from utils.taxi import curise_region
from utils.system_estimate import traveling_time_est, trans_prob_manipulation
from utils.flexibility_prediction.potential_policy import potential_opt



def TES_policy(parameter, J_trans, J_idle, P, initial_state, constant=None, date='2015-01-01'):
    data_path = os.path.join(ROOT,'data','taxitripperslot',date)
    
    parameter = parameter.copy()
    parameter['t_hat'] = int(np.floor(parameter['t_hat']/10))
    parameter['t_start'] = int(np.floor(parameter['t_start']/10))
    parameter['T_b'] = int(np.floor(parameter['T_b']/10))
    parameter['tau'] = int(np.floor(parameter['tau']/10))
    parameter['T_c'] = int(np.floor(parameter['T_c']/10))
    if "slot_length" in parameter.keys():
        parameter["slot_length"] = int(np.floor(parameter['slot_length']/10))
    else:
        parameter["slot_length"] = 1 # 10 min

    if constant is None:
        constant = {
            "L":30,
            "L2":6,
            "L_threshold":4,
            "num_of_taxis":13000, # 13000
            "num_of_regions":38
        }

    baseline_metric = {
        "J_trans":J_trans,
        "J_idle":J_idle,
        "P":P
    }

    decision, metric, state = TES_opt(parameter,initial_state,baseline_metric,constant,data_path)
    return decision, metric, state

def TES_opt(parameter,initial_state,baseline_metric,constant,demand_data_path):
    data_path = os.path.join(ROOT, 'data')

    slot_length = parameter["slot_length"]
    t_hat = min(144-parameter["t_start"],parameter["t_hat"])
    T_c = min(144-parameter["t_start"]-parameter["t_hat"],parameter["T_c"])
    T_c = int(np.ceil(T_c/slot_length))
    tau = min(144-parameter["t_start"]-parameter["t_hat"],parameter["tau"])
    tau = int(np.ceil(tau/slot_length))
    T_b = min(144-parameter["t_start"]-parameter["t_hat"]-parameter["tau"],parameter["T_b"])
    T_b = int(np.ceil(T_b/slot_length))

    # parameter["slot_length"] = slot_length
    parameter["t_hat"] = t_hat
    parameter["T_c"] = T_c
    parameter["T_b"] = T_b
    parameter["tau"] = tau
        

    
    #region: load data
    # load charging slot data
    e=[] # maximum number of taxis can be charged in each region during time slot t, e^t_j in paper
    for slot in range(T_c):
        with open(os.path.join(data_path,"pointsnum_per_region")) as f:
        # with open(currentdatepath+str(t_start+slot)+'_etaxicapacity') as f:
            region =[]
            for line in f:
                line= line.strip('\n')
                line = float(line)
                region.append(line)
            e.append(region)

    reachlist=[]
    travelingTime = np.zeros((constant["num_of_regions"],constant["num_of_regions"]),dtype="int")
    for region in range(constant["num_of_regions"]):
        travelingTime[region,:], one = traveling_time_est(region,constant=constant,slot_length=parameter["slot_length"])
        reachlist.append(one)

    # c=[]
    # fopen = open(os.path.join(data_path, 'reachable'),'r')
    # for k in fopen:
    #     k=k.strip('\n')
    #     k=k.split(',')
    #     one =[]
    #     for value in k:
    #         one.append(int(float(value)))
    #     c.append(one)
    # for region in range(constant["num_of_regions"]):
    #     one=[]
    #     for j in range(constant["num_of_regions"]):
    #         if c[region][j]==0:
    #             one.append(j)
    #     reachlist.append(one)


    # load demand data and transition matrix
    transition_new = np.zeros((T_c,constant["num_of_regions"],constant["num_of_regions"])) # transition matrix using demand data
    estimatedpassengerdemand = np.zeros((constant["num_of_regions"],T_c)) # estimated passenger demand
    for slot in range(T_c):
        current_demand=np.zeros((constant["num_of_regions"],constant["num_of_regions"]))
        for slot_sub in range(slot_length):
            tmp = []
            with open(os.path.join(demand_data_path,str(parameter["t_start"]+int(slot*slot_length)+slot_sub)+'_demand'),'r') as f:
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


    """transition_new=[] # ???: mobility pattern
    for slot in range(T_c):
        with open(os.path.join(data_path,'transition', str(slot+parameter["t_start"]+t_hat))) as f:
            data=[]
            for line in f:
                line = line.strip('\n')
                line= line.split(',')
                one=[]
                for k in line:
                    one.append(float(k))
                data.append(one)
            transition_new.append(data)"""

    transition_man = np.zeros((transition_new.shape[0],constant["num_of_regions"],constant["num_of_regions"],constant["L"]))
    for t in range(transition_man.shape[0]):
        transition_man[t,:,:,:] = trans_prob_manipulation(transition=transition_new[t,:,:],travelingTime=travelingTime,constant=constant)
    transition = transition_man

    travelingdistance =[]
    fopen = open(os.path.join(data_path, 'travelingdistance_region'),'r')
    for k in fopen:
        k=k.strip('\n')
        k=k.split(',')
        one =[]
        for value in k:
            one.append((float(value)))
        travelingdistance.append(one)
    fopen.close()

    transition_constants = {
        "transition":transition,
        "travelingTime":travelingTime,
        "travelingdistance":travelingdistance,
        "reachlist":reachlist,
        "estimatedpassengerdemand":estimatedpassengerdemand,
        "e":e
    }
    #endregion

    decision, metric, state = TES_opt_stage1(parameter,initial_state,baseline_metric,constant,transition_constants)

    return decision,metric,state

def TES_opt_stage1(parameter,initial_state,baseline_metric,constant,transition_constants):
    
    slot_length = parameter["slot_length"]
    t_hat = parameter["t_hat"]
    T_c = parameter["T_c"]
    tau = parameter["tau"]
    T_b = parameter["T_b"]

    transition = transition_constants["transition"]
    travelingTime = transition_constants["travelingTime"]
    travelingdistance = transition_constants["travelingdistance"]
    reachlist = transition_constants["reachlist"]
    estimatedpassengerdemand = transition_constants["estimatedpassengerdemand"]
    e = transition_constants["e"]


    
    
    # try:
    m = Model("CPS")

    #region: Create variables
    x = {}  # ???: number of unoccupied working taxis in region i dispatched to region j's charging station
    xd = {} # ???: number of taxis in region i's charging station dispatched to region j's charging station.
    y = {}  # ???: number of unoccupied taxi in region i dispatched to region j
    yd = {} # ???: number of taxis in region i's charging station dispatched to region j

    # initialize variables
    for i in range(constant["num_of_regions"]):
        for j in reachlist[i]:
            for l in range(constant["L"]):
                for t in range(T_c):
                    x[i,j,l,t] = m.addVar(lb=0.0, ub=constant['num_of_taxis'], vtype=GRB.CONTINUOUS, name="xp[%s,%s,%s,%s]" % (i,j,l,t))
                    y[i,j,l,t] = m.addVar(lb=0.0, ub=constant['num_of_taxis'], vtype=GRB.CONTINUOUS, name="yp[%s,%s,%s,%s]" % (i, j, l, t))

                    xd[i,j,l,t] = m.addVar(lb=0.0, ub=constant['num_of_taxis'], vtype=GRB.CONTINUOUS, name="xd[%s,%s,%s,%s]" % (i, j, l, t))
                    yd[i,j,l,t] = m.addVar(lb=0.0, ub=constant['num_of_taxis'], vtype=GRB.CONTINUOUS, name="yd[%s,%s,%s,%s]" % (i, j, l, t))
    
    u={} # ???: number of taxis charging in region i during time slot t, u^{l,t}_{j} in paper
    for i in range(constant["num_of_regions"]):
        for l in range(constant["L"]):
            for t in range(T_c):
                u[i,l,t] = m.addVar(lb= 0.0, vtype= GRB.CONTINUOUS, name="u[%s,%s,%s]" %(i,l,t))

    served={}
    for i in range(constant["num_of_regions"]):
        for t in range(T_c):
            served[i,t] = m.addVar(lb=0.0, ub=constant['num_of_taxis'],vtype=GRB.CONTINUOUS, name="served[%s,%s]" % (i,t))

    #endregion

    #region: create intermediate variables
    S={} # number of taxi supply
    V={} # number of vacant taxis
    D={} # number of taxis in the charging stations
    O={} # number of occupied taxis
    # initialize expressions
    for i in range(constant["num_of_regions"]):
        for t in range(T_c+1):
            for l in range(constant["L"]):
                V[i,l,t] = LinExpr()
                D[i,l,t] = LinExpr()
                O[i,l,t] = LinExpr()

                if t < T_c:
                    S[i,l,t] = LinExpr(sum( (y[j,i,l,t]+yd[j,i,l,t]) for j in reachlist[i] ))
                """
                if l==0:
                    S[i,l,t] = LinExpr(sum( (y[j,i,l,t]+yd[j,i,l,t]+y[j,i,l+1,t]+yd[j,i,l+1,t]) for j in reachlist_ex[i] ))
                    S[i,l,t].add(y[i,i,l,t]+yd[i,i,l,t])
                elif l < constant["L"]-1:
                    S[i,l,t] = LinExpr(sum( (y[j,i,l+1,t]+yd[j,i,l+1,t]) for j in reachlist_ex[i] ))
                    S[i,l,t].add(y[i,i,l,t]+yd[i,i,l,t])
                else:
                    S[i,l,t] = LinExpr(y[i,i,l,t]+yd[i,i,l,t])
                """

    #######################################################
    for i in range(constant["num_of_regions"]):
        for l in range(constant["L"]):
            for t in range(1,initial_state["occupied"].shape[2]):
                for tt in range(max(t-T_c,1),t+1):
                    O[i,min(l+tt,constant["L"]-1),t-tt].add(initial_state["occupied"][i,l,t])
                pass
    #######################################################

    for i in range(constant["num_of_regions"]):
        for l in range(constant["L"]):
            V[i,l,0].add(initial_state["vacant"][i][l])
            D[i,l,0].add(initial_state["waiting"][i][l]+initial_state["charging"][i][l])

            for t in range(T_c):
                # if  t < initial_state["occupied"].shape[2]: 
                if t >= 1 and t < initial_state["occupied"].shape[2]:
                    V[i,l,t].add(initial_state["occupied"][i,l,t]) # turn initially occupied taxis to vacant

                if l==0:
                    pass # vacant taxi with empty battery does not exist
                else:
                    for j in range(constant['num_of_regions']):
                        if j in reachlist[i]:
                            if l >= 1:
                                V[i,l-1,t+1].add(transition[t][j][i][l]*S[j,l,t]) # POET equation (3)
                            else:
                                V[i,l,t+1].add(transition[t][j][i][l]*S[j,l,t]) # POET equation (3)
                        else:
                            k = travelingTime[j,i]
                            if k+t<T_c: # arrival time within control horizon
                                if l>=k:
                                    V[i,l-k,t+k].add(transition[t][j][i][l]*S[j,l,t])
                                else:
                                    V[i,0,t+k].add(transition[t][j][i][l]*S[j,l,t])
                            
                            #######################################################################
                            for tt in range(1,k):
                                if tt+t < T_c+1 and transition[t][j][i][l]>0:
                                    O[i,l-tt,t+tt].add(transition[t][j][i][l]*S[j,l,t])
                                pass
                            ########################################################################
                
                # POET equation in the end of section 5.3
                # xp and xd decision won't consume energy
                if l < constant["L2"]:
                    D[i,l,t+1].add(sum( (x[j,i,l,t]+xd[j,i,l,t]) for j in reachlist[i]) -u[i,l,t]) # u[i,l-L2,t-1]=0
                elif l <constant["L"]-1:
                    D[i,l,t+1].add(u[i,l-constant["L2"],t] + sum( (x[j,i,l,t]+xd[j,i,l,t]) for j in reachlist[i]) -u[i,l,t])
                else:
                    D[i,l,t+1].add(
                        sum((u[i,ll,t]) for ll in np.arange(constant["L"]-constant["L2"]-1,constant["L"]-1))
                        +sum( (x[j,i,l,t]+xd[j,i,l,t]) for j in reachlist[i])
                    )
                
                """# xp and xd decisions will consume energy
                if l == 0:
                    D[i,l,t+1].add(
                        sum( (x[j,i,l,t]+xd[j,i,l,t]+x[j,i,l+1,t]+xd[j,i,l+1,t]) for j in reachlist_ex[i]) 
                        + x[i,i,l,t]+xd[i,i,l,t]
                        - u[i,l,t]
                    ) # u[i,l-L2,t-1]=0
                elif l < constant["L2"]:
                    D[i,l,t+1].add(
                        sum( (x[j,i,l+1,t]+xd[j,i,l+1,t]) for j in reachlist_ex[i])
                        + x[i,i,l,t]+xd[i,i,l,t]
                        - u[i,l,t]
                    ) # u[i,l-L2,t-1]=0
                elif l <constant["L"]-1:
                    D[i,l,t+1].add(
                        u[i,l-constant["L2"],t] 
                        + sum( (x[j,i,l+1,t]+xd[j,i,l+1,t]) for j in reachlist_ex[i])
                        + x[i,i,l,t]+xd[i,i,l,t]
                        - u[i,l,t])
                else: # l == L-1, no taxis are dispatched to charging station
                    D[i,l,t+1].add(
                        sum((u[i,ll,t]) for ll in np.arange(constant["L"]-constant["L2"]-1,constant["L"]-1))
                        + x[i,i,l,t]+xd[i,i,l,t]
                    )"""
            
            if T_c < initial_state["occupied"].shape[2]:
                V[i,l,T_c].add(initial_state["occupied"][i,l,T_c]) # turn initially occupied taxis to vacant
    #endregion

    #region: create constraints
    for i in range(constant["num_of_regions"]):
        for l in range(constant["L"]):
            for t in range(T_c):
                m.addConstr(V[i,l,t] == sum((x[i,j,l,t]+y[i,j,l,t]) for j in reachlist[i])) # equation (1)
                m.addConstr(D[i,l,t] == sum((xd[i,j,l,t] + yd[i,j,l,t]) for j in reachlist[i])) # equation (1)

    for i in range(constant["num_of_regions"]):
        for l in range(constant["L"]):
            for t in range(T_c):
                m.addConstr( u[i,l,t] <= sum( (x[j,i,l,t]+xd[j,i,l,t] ) for j in reachlist[i] )) # equation (6)

                """# decision xp and xd will consume energy
                if l == 0:
                    m.addConstr( u[i,l,t] <= 
                        sum( (x[j,i,l,t]+xd[j,i,l,t]+x[j,i,l+1,t]+xd[j,i,l+1,t] ) for j in reachlist_ex[i] )
                        + x[i,i,l,t]+xd[i,i,l,t]
                    ) # equation (6)
                    pass
                elif l < constant["L"]-1:
                    m.addConstr( u[i,l,t] <= 
                        sum( (x[j,i,l+1,t]+xd[j,i,l+1,t] ) for j in reachlist_ex[i] )
                        + x[i,i,l,t]+xd[i,i,l,t]
                    ) # equation (6)
                else:
                    # m.addConstr(u[i,l,t] <= sum( (x[j,i,l,t]+xd[j,i,l,t] ) for j in reachlist[i] ))
                    m.addConstr(u[i,l,t] <= x[i,i,l,t]+xd[i,i,l,t])"""
    
    for i in range(constant["num_of_regions"]):
        for t in range(T_c):
            m.addConstr(sum(u[i, l, t] for l in range(constant["L"])) <= e[t][i]) # equation (6)

    for i in range(constant["num_of_regions"]):
        for l in range(constant["L_threshold"]):
            for t in range(T_c):
                m.addConstr(S[i,l,t]==0) # taxis with battery energy lower than L_threshold should go for charging


    for i in range(constant["num_of_regions"]):
        for j in reachlist[i]:
            for l in range(constant["L"]-1,constant["L"]):
                for t in range(T_c):
                    # if l>10: # ???: what does 10 represents?
                    m.addConstr(x[i,j,l,t] ==0 )    # high battery taxis don't charge
                    m.addConstr(xd[i,j,l,t]  == 0)

    

    for i in range(constant["num_of_regions"]):
        for t in range(T_c):
            m.addConstr( served[i,t]<= estimatedpassengerdemand[i][t] ) # number of served passenger = min(passenger demand, number of vacant taxis)
            m.addConstr( served[i,t]<= sum(S[i,l,t] for l in range(constant["L"])) )

    
    # charging constraint (2e)
    if parameter["charging_constraint"]:
        for t in range(tau):
            m.addConstr(
                0 <= sum(
                    sum(
                        (baseline_metric["P"][i,l,t]-u[i,l,t]) 
                    for l in range(constant['L'])) 
                for i in range(constant['num_of_regions'])) # type: ignore
            )

    
    J_trans = LinExpr(
        sum(
            sum(
                (served[i,t]) 
            for i in range(constant["num_of_regions"])) 
        for t in range(T_c))
    )
    ################## Constraints for tolerance of passenger loss ###################
    # # OPTION 1: J_trans constraint (2f)
    # J_trans_baseline = sum(
    #     sum(
    #         (baseline_metric["J_trans"][i,t]) 
    #     for i in range(constant["num_of_regions"])) 
    # for t in range(T_c))
    # m.addConstr(J_trans >= (1-parameter['epsilon']) * J_trans_baseline)

    # OPTION 2: passenger tolerance for every time slot
    J_trans_baseline = np.sum(baseline_metric["J_trans"],0)

    for t in range(tau):
        m.addConstr(
            sum(
                (served[i,t]) 
            for i in range(constant["num_of_regions"])) 
        >= J_trans_baseline[t]
    )

    for t in range(tau,T_c):
        m.addConstr(
            sum(
                (served[i,t]) 
            for i in range(constant["num_of_regions"])) 
        >= (1-parameter['epsilon']) * J_trans_baseline[t]
    )
    
    # # OPTION 3: passenger tolerance for every time slot every regions
    # for t in range(T_c):
    #     for i in range(constant["num_of_regions"]):
    #         m.addConstr(served[i,t] >= (1-parameter["epsilon"])*baseline_metric["J_trans"][i,t])

    ##################################################################################

    # idle driving distance constraint (2g)
    if ~np.isinf(parameter['ita']):
        m.addConstr(
            sum(
                sum(
                    sum(
                        sum(
                            ((x[i,j,l,t]+xd[i,j,l,t]+y[i,j,l,t]+yd[i,j,l,t])* travelingdistance[i][j]) 
                        for j in reachlist[i]) 
                    for i in range(constant['num_of_regions'])) 
                for l in range(constant['L'])) 
            for t in range(tau))
            <= (1+parameter['ita']) * baseline_metric['J_idle']
        )

    # # Power reduction constraint (2d), ENABLE this part when guarantee power reduction is used to measuere flexibility
    # Delta_P = m.addVar(lb=0,ub=constant['num_of_taxis'], vtype=GRB.CONTINUOUS, name="Delta_P")
    # for t in range(min(tau,T_c), min(tau+T_b,T_c)):
    #     m.addConstr(
    #         Delta_P <= sum(
    #             sum(
    #                 (baseline_metric["P"][i,l,t]-u[i,l,t])
    #             for l in range(constant["L"])) 
    #         for i in range(constant['num_of_regions']))
    #     )

    # obj_trans = sum(sum( served[i,t] for i in range(constant["num_of_regions"]) ) for t in range(T_c))

    #endregion
    
    # objective
    obj_charging = LinExpr()

    # ## behavior 1
    # for l in range(constant["L"]):
    #     for i in range(constant["num_of_regions"]):
    #         objective.add(l*V[i,l,tau]+l*D[i,l,tau]) # energy storage

    ## maximize charging
    for i in range(constant["num_of_regions"]):
        for l in range(constant["L"]):
            for t in range(tau):
                obj_charging.add(u[i,l,t])

    beta = 1e-4
    objective = obj_charging + beta*J_trans # also add service quality in the future as objective

    m.setObjective(objective,GRB.MAXIMIZE)

    m.Params.BarHomogeneous = 1
    # m.setParam("OptimalityTol",0.01)
    m.setParam("Quad",0)
    m.setParam("Method",2)
    m.setParam("MIPFocus",3)
    m.setParam("Presolve",2)
    m.setParam("NodeMethod",2)
    m.setParam("NoRelHeurWork",100)
    
    
    # m.feasRelax(relaxobjtype=0,minrelax=False,vars=None,lbpen=None,ubpen=None,constrs=slack_constr,rhspen=[10]*len(slack_constr))
    # m.computeIIS()

    m.optimize()

    xp_val = np.zeros((constant["num_of_regions"],constant["num_of_regions"],constant["L"],tau))
    xd_val = np.zeros((constant["num_of_regions"],constant["num_of_regions"],constant["L"],tau))
    yp_val = np.zeros((constant["num_of_regions"],constant["num_of_regions"],constant["L"],tau))
    yd_val = np.zeros((constant["num_of_regions"],constant["num_of_regions"],constant["L"],tau))
    served_val = np.zeros((constant["num_of_regions"],tau))
    u_val = np.zeros((constant["num_of_regions"],constant["L"],tau))

    for i in range(constant["num_of_regions"]):
        for j in reachlist[i]:
            for l in range(constant["L"]):
                for t in range(tau):
                    xp_val[i,j,l,t] = x[i,j,l,t].x
                    xd_val[i,j,l,t] = xd[i,j,l,t].x
                    yp_val[i,j,l,t] = y[i,j,l,t].x
                    yd_val[i,j,l,t] = yd[i,j,l,t].x
    for i in range(constant["num_of_regions"]):
        for l in range(constant["L"]):
            for t in range(tau):
                u_val[i,l,t] = u[i,l,t].x
    for i in range(constant["num_of_regions"]):
        for t in range(tau):
            served_val[i,t] = served[i,t].x

    vacant = np.zeros((constant["num_of_regions"],constant["L"],tau+1))
    charging = np.zeros((constant["num_of_regions"],constant["L"],tau+1))
    waiting = np.zeros((constant["num_of_regions"],constant["L"],tau+1))
    supply = np.zeros((constant["num_of_regions"],constant["L"],tau+1))
    for i in range(constant['num_of_regions']):
        for l in range(constant['L']):
            # vacant[i,l,0] = np.round(V[i,l,0].getValue())
            # supply[i,l,0] = np.round(S[i,l,0].getValue())
            vacant[i,l,0] = V[i,l,0].getValue()
            supply[i,l,0] = S[i,l,0].getValue()

            charging[i,l,0] = initial_state['charging'][i][l]
            waiting[i,l,0] = initial_state['waiting'][i][l]
            

            for t in range(1,tau+1):
                # vacant[i,l,t] = np.round(V[i,l,t].getValue())
                # supply[i,l,t] = np.round(S[i,l,t].getValue())
                # waiting[i,l,t] = np.round(sum( (x[j,i,l,t-1].x + xd[j,i,l,t-1].x ) for j in reachlist[i] )-u[i,l,t-1].x) # take round after calculation
                vacant[i,l,t] = V[i,l,t].getValue()

                if t < tau:
                    supply[i,l,t] = S[i,l,t].getValue()

                if l< constant['L2']:
                    charging[i,l,t] = 0
                elif l<constant['L']-1:
                    # charging[i,l,t] = np.round(u[i,l-constant['L2'],t-1].x)
                    charging[i,l,t] = u[i,l-constant['L2'],t-1].x
                else:
                    # charging[i,l,t] = sum(np.round(u[i,ll,t-1].x) for ll in range(constant["L"]-constant["L2"],constant["L"]))
                    charging[i,l,t] = sum(u[i,ll,t-1].x for ll in range(constant["L"]-constant["L2"]-1,constant["L"]-1)) 


                waiting[i,l,t] = sum( (x[j,i,l,t-1].x + xd[j,i,l,t-1].x ) for j in reachlist[i] )-u[i,l,t-1].x
                
                """if l == 0:
                    waiting[i,l,t] = sum( (x[j,i,l,t-1].x + xd[j,i,l,t-1].x + x[j,i,l+1,t-1].x + xd[j,i,l+1,t-1].x ) for j in reachlist_ex[i] ) \
                        + x[i,i,l,t-1].x + xd[i,i,l,t-1].x \
                        - u[i,l,t-1].x
                elif l<constant['L']-1:
                    waiting[i,l,t] = sum( (x[j,i,l,t-1].x + xd[j,i,l,t-1].x) for j in reachlist_ex[i] ) \
                        + x[i,i,l,t-1].x + xd[i,i,l,t-1].x \
                        - u[i,l,t-1].x
                else:
                    waiting[i,l,t] = x[i,i,l,t-1].x + xd[i,i,l,t-1].x"""
                
    occupied = np.zeros((constant["num_of_regions"],constant["L"],tau+1))
    for i in range(constant['num_of_regions']):
        for l in range(constant['L']):
            for t in range(tau+1):
                occupied[i,l,t] = O[i,l,t].getValue()

    obj_val = m.ObjVal

    decision = {"xp":xp_val,"xd":xd_val,"yp":yp_val,"yd":yd_val,"u":u_val}
    metric = {"num_charging":u_val,"J_trans":served_val,"objective":obj_val}
    state = {"vacant":vacant,"charging":charging,"waiting":waiting,"supply":supply,"estimated_demand":estimatedpassengerdemand,"occupied":occupied}

    J_idle = np.zeros(tau)
    for t in range(tau):
        for l in range(constant['L']):
            for i in range(constant['num_of_regions']):
                for j in reachlist[i]:
                    J_idle[t] += (xp_val[i,j,l,t]+xd_val[i,j,l,t]+yp_val[i,j,l,t]+yp_val[i,j,l,t]) * travelingdistance[i][j]

    metric['J_idle'] = J_idle

    # deltaP_val = Delta_P.x
    # metric['Delta_P'] = deltaP_val

    # except:
    #     print("*******************************FAILED*******************************")
    #     decision = None
    #     metric = None
    #     state = None

    return decision, metric, state

def TES_policy_mpc(parameter:dict, constant:dict, J_trans_baseline:np.ndarray, J_idle_baseline:np.ndarray, P_baseline:np.ndarray, initial_state, date:str='2015-01-01',log:bool=False):
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
            c_parameter = {
                "t_start":time,
                "t_hat":0,
                "T_b":max(0,parameter["tau"]-(t*parameter["slot_length"])),
                # "tau":max(0,parameter["tau"]-(t*parameter["slot_length"])),
                "tau":parameter["slot_length"],
                "T_c":(t_end-t)*parameter["slot_length"],
                "slot_length":parameter["slot_length"],
                "epsilon":parameter["epsilon"],
                "ita":parameter["ita"],
                "charging_constraint":parameter["charging_constraint"]
            }

            decision_new, metric_new, state_new = TES_opt(
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

        num_served = metric["J_trans"][:,counter]

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
        print()
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
        "J_idle":J_idle
    }

    return decision_overall, metric_overall, state_overall, taxi_state
   