
import numpy as np
import pickle




def Sort_Columns(matrix):
    # Convert the input matrix to a numpy array for easier manipulation
    matrix = np.array(matrix)
    
    # Initialize a list to store the sorting criteria for each column
    criteria = []
    
    # Iterate over each column to calculate the criteria
    for col_idx in range(matrix.shape[1]):
        col = matrix[:, col_idx]
        max_row_index = np.max(np.where(col == 1)[0]) if 1 in col else -1
        sum_row_indices = np.sum(np.where(col == 1)[0])
        criteria.append((max_row_index, sum_row_indices, col_idx))
    
    # Sort the criteria list based on the tuple values
    criteria.sort(key=lambda x: (x[0], x[1]))
    
    # Extract the sorted column indices
    sorted_col_indices = [col_idx for _, _, col_idx in criteria]
    
    # Rearrange the columns of the original matrix based on the sorted column indices
    sorted_matrix = matrix[:, sorted_col_indices]
    
    return np.array(sorted_matrix.tolist())


# Example usage
# matrix = [[1, 0, 0, 1],
#           [0, 1, 0, 1],
#           [0, 0, 1, 1],
#           [1, 1, 1, 0]]
# sorted_matrix = sort_columns(matrix)

def get_X(varInfo, m, n):
    X = np.zeros((m, n))
    for var in varInfo:
        if var[0].startswith('x'):
            temp = var[0][1:].strip('][').split(',')
            i = int(temp[0].strip('message'))-1
            j = int(temp[1].strip('tag'))-1
            X[i, j]  = np.round(var[1],0)
    return X

# Example usage
# varInfo = [('x[0,0]', 1.0), ('x[0,1]', 0.0), ('x[0,2]', 0.0), ('x[1,0]', 1.0), ('x[1,1]', 0.0), ('x[1,2]', 0.0), ('x[2,0]', 1.0), ('x[2,1]', 0.0), ('x[2,2]', 0.0), ('x[3,0]', 1.0), ('x[3,1]', 0.0), ('x[3,2]', 0.0)]
# m = 4
# n = 3
# X = get_X(varInfo, m, n)
# print(X)

#check  if the experiment is already saved
def Check_Experiment(parameters, filePath = 'Xs.pkl'):
    try:
        experiments = Load_Experiments(filePath)
    except:
        experiments = {}
    for i in range(len(experiments)):
        try:
            if experiments[i]['parameters'] == parameters:
                print("Experiment with the same parameters already exist as experiment number",i)
                return experiments[i]
        except:
            pass
    return None

def Save_Experiment(experiment, filePath = 'Xs.pkl'):
    try:
        with open(filePath, 'rb') as f:
            experiments = pickle.load(f)
            experiment_nr = len(experiments)
            experiments[experiment_nr]=experiment
    except:
        experiment_nr = 0
        experiments = {experiment_nr: experiment}

    #check if the experiment is already saved
    if Check_Experiment(parameters=experiment['parameters']) is not None:
        return
    
    with open(filePath, 'wb') as f:
        pickle.dump(experiments, f)
    print("Experiment saved as experiment number",experiment_nr)

def Load_Experiments(filePath = 'Xs.pkl'):
    import pickle
    with open(filePath, 'rb') as f:
        return pickle.load(f)


def Run_Experiment(model, parameters:dict, eval, m_size, t_size, save:bool = True):

    experiment = Check_Experiment(parameters=parameters)
    if experiment is not None:
        return experiment

    varInfo = model(**parameters)

    X = get_X(varInfo, parameters['m_nr'], parameters['t_nr'])
    X = Sort_Columns(X) 

    results = {'varInfo': varInfo, 'X': X, 'Y': Get_Y(X)} 
    experiment = {'parameters': parameters, 'results': results}


    experiment['eval'] = eval(experiment, m_size, t_size, plot = False)

    
    # save the experiment
    if save:
        Save_Experiment(experiment = experiment)

    return experiment



def find_augmenting_path(X, u, match_from_V2_to_V1, visited):
    for v in range(len(X[0])):
        if X[u][v] == 1 and not visited[v]:
            visited[v] = True
            if match_from_V2_to_V1[v] == -1 or find_augmenting_path(X, match_from_V2_to_V1[v], match_from_V2_to_V1, visited):
                match_from_V2_to_V1[v] = u
                return True
    return False

def Get_Y(X):
    # Number of vertices in V1 and V2
    V1_size = len(X)
    V2_size = len(X[0])

    # Array to store the match from V2 to V1, initialized to -1 (no match)
    match_from_V2_to_V1 = [-1] * V2_size

    # Try to find a match for every node in V1
    for u in range(V1_size):
        visited = [False] * V2_size  # Keeps track of visited nodes in V2 for each attempt
        find_augmenting_path(X, u, match_from_V2_to_V1, visited)

    # Check if we found a right-perfect matching
    if all(x != -1 for x in match_from_V2_to_V1):
        # Create the result matrix Y
        Y = np.zeros_like(X)
        for j in range(V2_size):
            Y[match_from_V2_to_V1[j]][j] = 1
        return Y
    else:
        return None