#import numpy as np
#import pandas as pd
#from scipy.stats import binom
import gurobipy as gp
from gurobipy import *
import csv


def math_model(M_size=10, T_size=10, p=1, q=1, TagEveryMessage=True, AtLeastOnce=False):
    
    # Sets
    I = ['message'+str(i+1) for i in range(M_size)]
    J = ['tag'+str(j+1) for j in range(T_size)]
    #A = ['aggregate'+str(i) for i in range(M_size+1)]
    K = [i for i in range(M_size+1)]
    
    # Parameters
    w = {}
    for j in J:
        for k in K:
        #w{(k, j): (q * (p**k))}
            if k == 0:
                w[(j, k)]=0
            else:    
                w[(j, k)]=q * (p**k)
    
    try:
        m = gp.Model("TagModel")
        # m.setParam('OutputFlag', 0)
        m.setParam(GRB.Param.TimeLimit, 21600)
        #m.setParam(GRB.Param.MIPGap, 0.01)
        #m.setParam(GRB.Param.MIPFocus, 2)
        
        #Create Variables
        
        x = m.addVars(I, J, vtype=GRB.BINARY, name="x")
        z = m.addVars(J, K, vtype=GRB.BINARY, name="z")
        y = m.addVars(I, J, K, vtype=GRB.BINARY, name="y")
        
        if(AtLeastOnce==True):
            A = m.addVars(I, J, K, name="A")
            #w = m.addVars(I, len(J)-1, name="w")
            q = m.addVars(I, len(J), name="q")
            
        
        # Model Obj
        #m.setObjective(
        #    gp.quicksum(y[i, j, k] * w[j, k] for i in I for j in J for k in K),
        #    GRB.MAXIMIZE
        #)
        
        if(AtLeastOnce==True):
            m.setObjective(
                gp.quicksum((1 - q[i,len(J)-1]) for i in I),
                GRB.MAXIMIZE
            )
        else:
            m.setObjective(
                gp.quicksum(y[i, j, k] * w[j, k] for i in I for j in J for k in K),
                GRB.MAXIMIZE
            )
            
        
        # Constraints
        
        m.addConstrs((gp.quicksum(x[i, j] for i in I) == gp.quicksum(k*z[j, k] for k in K) for j in J), name='AggCount')
        
        m.addConstrs((gp.quicksum(z[j, k] for k in K) <= 1 for j in J), name='Select_Single_Agg')
        
        m.addConstrs((x[i, j] + z[j, k] >= 2*y[i, j, k] for i in I for j in J for k in K), name='Assign_Y')
        
        
        if(TagEveryMessage==True):
            m.addConstrs((gp.quicksum(x[i, j] for j in J) >= 1 for i in I), name='Tag_Every_Message')
        
        
        if(AtLeastOnce==True):
            m.addConstrs(((y[i, j, k] * w[j, k] == A[i,j,k]) for i in I for j in J for k in K), name='Auth_Value')
            m.addConstrs(((1-gp.quicksum(A[i,'tag1',k] for k in K)) == q[i, 0] for i in I) , name='Auxiliary_Var_Init')
            for j in range(1, len(J)):
                m.addConstrs((q[i,j-1]*(1-gp.quicksum(A[i,J[j],k] for k in K)) == q[i,j] for i in I), name='Auxiliary_Vars')
                
        
        #m.addConstr((gp.quicksum(y1[v] + y2[v] for v in V) <= 1), name='5d')
        #m.addConstrs((w[s] >= eta - Pi[s] for s in S), name='6c')
        
        m.write('TagModel.lp')
        m.optimize()
        
        if m.status == GRB.UNBOUNDED:
            print('The model is unbounded')
        #if m.status == GRB.OPTIMAL:
            #print('The optimal objective is %g' % m.objVal)
        if m.status == GRB.INF_OR_UNBD or m.status == GRB.INFEASIBLE:
            print('The model is infeasible; computing IIS')
            m.computeIIS()
            if m.IISMinimal:
                print('IIS is minimal\n')
            else:
                print('IIS is not minimal\n')
            print('\n The following constraint(s) cannot be satisfied')
            for c in m.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)
                    
        varInfo = [(v.varName, v.X) for v in m.getVars() if v.X>0]
        obj = ('Obj', m.objVal)
        varInfo.append(obj)
        
        #with open('TagResult_'+str(M_size)+'_'+str(T_size)+'.csv', 'w', newline='') as myfile:    
        #    w = csv.writer(myfile, delimiter=' ')
        #    w.writerows(varInfo)
        return varInfo
        
    except gurobipy.GurobiError as e:
        print('Error code '+ str(e.errno) + ': ' + str(e))
    
    #except AttributeError:
    #    print('Encountered an attribute error')    