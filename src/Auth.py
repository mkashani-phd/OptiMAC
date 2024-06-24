import itertools
import numpy as np
import pandas as pd
import pickle
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD

            #  t1  t2  t3  t4  t5  t6  t7  t8  t9
X = np.array([[ 1,  0,  0,  0,  0,  0,  1,  0,  0], # m1
              [ 1,  0,  0,  0,  0,  0,  0,  1,  0], # m2
              [ 1,  0,  0,  0,  0,  0,  0,  0,  1], # m3
              [ 0,  1,  0,  0,  0,  0,  1,  0,  0], # m4
              [ 0,  1,  0,  0,  0,  0,  0,  1,  0], # m5
              [ 0,  1,  0,  0,  0,  0,  0,  0,  1], # m6
              [ 0,  0,  1,  0,  0,  0,  1,  0,  0], # m7
              [ 0,  0,  1,  0,  0,  0,  0,  1,  0], # m8
              [ 0,  0,  1,  0,  0,  0,  0,  0,  1], # m9
              [ 0,  0,  0,  1,  0,  0,  1,  0,  0], # m10
              [ 0,  0,  0,  1,  0,  0,  0,  1,  0], # m11
              [ 0,  0,  0,  1,  0,  0,  0,  0,  1], # m12
              [ 0,  0,  0,  0,  1,  0,  1,  0,  0], # m13
              [ 0,  0,  0,  0,  1,  0,  0,  1,  0], # m14
              [ 0,  0,  0,  0,  1,  0,  0,  0,  1], # m15
              [ 0,  0,  0,  0,  0,  1,  1,  0,  0], # m16
              [ 0,  0,  0,  0,  0,  1,  0,  1,  0], # m17
              [ 0,  0,  0,  0,  0,  1,  0,  0,  1]]) # m18

def ProMAC_experiment(parameters,x):
    M = parameters['m_nr']
    p = parameters['p']
    q = parameters['q']
    # Initialize an NxN matrix with zeros
    matrix = np.zeros((M, M), dtype=int)
    
    # Populate the matrix
    for i in range(M):
        # Determine the start position for the ones in the current row
        start_pos = i % M
        for j in range(x):
            col_pos = (start_pos + j) 
            if col_pos >= M:
                continue
            matrix[i, col_pos] = 1
    
    X =  np.array(matrix.tolist())
    return {'results': {'X': X}, 'parameters': {'m_nr': M, 't_nr': M, 'p': p, 'q': q}}

def random_binary_array(shape, probability_of_one=0.5):
    # Ensure p is between 0 and 1
    if not 0 <= probability_of_one <= 1:
        raise ValueError("Probability p must be between 0 and 1.")
    choices = [0, 1]
    probabilities = [1-probability_of_one, probability_of_one]
    # Generate the random array with the specified probabilities
    random_array = np.random.choice(choices, size=shape, p=probabilities)
    return random_array

# based on mask and multiply operation
def Validate(experiment,m,t,rectified = False, includeValidTag = False):
    X = experiment['results']['X']
    # mask and multiply
    m_nr,t_nr = X.shape
    mm = np.zeros(t_nr)
    for tag in range(t_nr):
        mask = m[np.where(X[:,tag] == 1)] # mask for tag
        mm[tag] = np.prod(mask) # multiply the mask
    A = np.matmul(X,(mm*t).transpose())

    A = np.array( [1 if x > 1 else x for x in A]) if rectified else A
    return (A, mm*t) if includeValidTag else A

# Wrong mathematics because the dependency is not considered
# def Probability(X,m,t):    
#     m_nr,t_nr = X.shape
#     mm = np.zeros(t_nr)
#     for tag in range(t_nr):
#         mask = m[np.where(X[:,tag] == 1)] # mask for tag
#         mm[tag] = np.prod(mask) # multiply the mask
#     U = (mm*t)
#     A = np.ones(m_nr)
#     for msg in range(m_nr):
#         for u in U[np.where(X[msg,:] == 1)]:
#             A[msg] *= (1-u)
#     return 1-A


def Latency(experiment,m,t,lost_penalty = 40):
    X = experiment['results']['X']
    m_nr,t_nr = X.shape
    A,validTags = Validate(experiment,m,t,includeValidTag=True)
    L = [np.where(X[:,np.where((X[msg,:]*validTags)>0)[0][0]] == 1)[0][-1]-msg if A[msg] >0 else lost_penalty for msg in range(m_nr)]

    return np.array(L)


def Reward(experiment,m,t,rectified_A = True, lost_penalty = 40, a = 1,l = 1, o = 100):
    X = experiment['results']['X']
    A = Validate(experiment,m,t,rectified=rectified_A)
    L= Latency(experiment,A,t,lost_penalty=lost_penalty)

    m_nr,t_nr = X.shape
    r = a*np.sum(A) - l*np.sum(L) - o * t_nr/m_nr

    return A, L, r



def Strength_Number_GPT(experiment):
    X = experiment['results']['X']
    X = np.array(X,dtype=int)

    m, n = X.shape
    for r in range(1, m+1):
        for rows in itertools.combinations(range(m), r):
            combined_row = np.zeros(n, dtype=int)
            for row in rows:
                combined_row += np.array(X[row])
            if all(combined_row >= 1):
                return np.array(list(rows))
    return np.array([])

def  Get_Strength_Number(experiment):
    X = experiment['results']['X']
    X = np.array(X,dtype=int)

    n_msg, n_tag = X.shape
    A = np.transpose(X)
    B = np.ones(n_tag)
    C = np.ones(n_msg)

    prob = LpProblem("Binary_LP_Problem", LpMinimize)

    # Define the variables
    x = [LpVariable(f'x{i}', cat='Binary') for i in range(len(C))]

    # Define the objective function
    prob += lpSum(C[i] * x[i] for i in range(len(C)))

    # Define the constraints
    for i in range(len(A)):
        prob += lpSum(A[i][j] * x[j] for j in range(len(C))) >= B[i]

    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=0))

    # Print the results
    print("Status:", prob.status)
    print("Objective value:", prob.objective.value())

    result = []
    for i in range(len(C)):
        #print(f"x{i+1}:", x[i].value())
        if x[i].value() == 1:
            result.append(i+1)
    return result

def Goodput(experiment,m,t,m_size=1024,t_size=256, tagAdjustment = False):
    X = experiment['results']['X']
    p = experiment['parameters']['p']
    q = experiment['parameters']['q']

    m_nr,t_nr = X.shape
    A = Validate(experiment,m,t)
    A = A>0
    if tagAdjustment:
        EA = np.average(Validate(experiment,np.array([p]*m_nr),np.array([q]*t_nr)))
        if EA < 1:
            EA = 1
        return (np.sum(A)*m_size)/(m_nr*m_size + int(t_size/EA)*t_nr)
    else:
        return (np.sum(A)*m_size)/(m_nr*m_size + t_nr*t_size)

def SecurityGoodput(experiment,m,t):
    m_nr,t_nr = experiment['parameters']['m_nr'],experiment['parameters']['t_nr']
    A = Validate(experiment,m,t)
    totalA = np.sum(Validate(experiment,np.ones(m_nr),np.ones(t_nr)))
    return np.sum(A)/totalA




# Example usage
# M = 6
# x = 3
# matrix = ProMAC_X(M, x)
# print(matrix)

def evaluate(experiment, plot = False):
    X = experiment['results']['X']
    p = experiment['parameters']['p']
    q = experiment['parameters']['q']

    X = np.array(X)
    m, n = X.shape
    if plot:
        import matplotlib.pyplot as plt
        plt.imshow(X, cmap= 'hot')
        plt.show()
    A = Validate(experiment, np.array([p]*m), np.array([q]*n))


    L = Latency(experiment, np.ones(m), np.ones(n))

    eval = {'A': A, 'L': L, 
            'average_A': np.average(A), 'average_L': np.average(L),
            'computation_(tag to message ratio)':n/m,
            'goodput_without_tag_adjustment': Goodput(experiment, np.ones(m), np.ones(n), tagAdjustment=False),
            'goodput_with_tag_adjustment': Goodput(experiment, np.ones(m), np.ones(n), tagAdjustment=True),
            'security_goodput': SecurityGoodput(experiment, np.array([p]*m), np.array([q]*n)),
            'rows_that_breaks_the_verification': Get_Strength_Number(experiment)}
    #pretty print the eval dictionary
    if plot:
        for key, value in eval.items():
            print(key, ' : ', value)

    return eval
    


    