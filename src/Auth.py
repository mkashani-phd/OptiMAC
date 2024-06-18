import itertools
import numpy as np

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

def random_binary_array(shape, probability_of_one=0.5):
    # Ensure p is between 0 and 1
    if not 0 <= probability_of_one <= 1:
        raise ValueError("Probability p must be between 0 and 1.")
    choices = [0, 1]
    probabilities = [1-probability_of_one, probability_of_one]
    # Generate the random array with the specified probabilities
    random_array = np.random.choice(choices, size=shape, p=probabilities)
    return random_array


########################## test ########################################
# Example usage
# shape = (18)  # Specify the desired shape of the array
# probability_of_one = 0.9  # Probability of 1 in the array

# # Generate the array
# random_array = random_binary_array(shape = shape,probability_of_one = probability_of_one)
# print(random_array)
#########################################################################


# based on closed form expression
def validate_slow(X,p,q):
    m_nr,t_nr = X.shape
    A = np.zeros(m_nr)
    for msg in range(m_nr):
        for t in range(t_nr):
            temp = 1
            for m in range(m_nr):
                temp *= X[m,t]*p[m] + 0**(X[m,t])
            A[msg] += X[msg,t]*q[t]*temp
    return A

# based on mask and multiply operation
def validate(X,m,t,rectified = False, includeValidTag = False):
    # mask and multiply
    m_nr,t_nr = X.shape
    mm = np.zeros(t_nr)
    for tag in range(t_nr):
        mask = m[np.where(X[:,tag] == 1)] # mask for tag
        mm[tag] = np.prod(mask) # multiply the mask
    A = np.matmul(X,(mm*t).transpose())

    A = np.array( [1 if x > 1 else x for x in A]) if rectified else A
    return (A, mm*t) if includeValidTag else A

def Latency(X,m,t,lost_penalty = 40):
    m_nr,t_nr = X.shape
    A,validTags = validate(X,m,t,includeValidTag=True)
    L = [np.where(X[:,np.where((X[msg,:]*validTags)>0)[0][0]] == 1)[0][-1]-msg if A[msg] >0 else lost_penalty for msg in range(m_nr)]

    return np.array(L)


def reward(X,m,t,rectified_A = True, lost_penalty = 40, a = 1,l = 1, o = 100):
    m_nr,t_nr = X.shape
    A = validate(X,m,t,rectified=rectified_A)
    L= Latency(X,A,t,lost_penalty=lost_penalty)
    r = a*np.sum(A) - l*np.sum(L) - o * t_nr/m_nr

    return A, L, r



def Strength_Number(matrix):
    m, n = len(matrix), len(matrix[0])
    for r in range(1, m+1):
        for rows in itertools.combinations(range(m), r):
            combined_row = np.zeros(n, dtype=int)
            for row in rows:
                combined_row += np.array(matrix[row])
            if all(combined_row >= 1):
                return np.array(list(rows))
    return np.array([])

######### test #########
# m_nr,t_nr = X.shape
# probability_of_success_message = .9
# probability_of_success_tag = 1
# m = random_binary_array(shape = m_nr, probability_of_one = probability_of_success_message)
# t = random_binary_array(shape = t_nr, probability_of_one = probability_of_success_tag)

# A = validate(X,m,t)

# print("m = ")
# print(m.reshape(6,-1))

# print("t = ")
# print(t)

# print("A = ")
# print(np.int8(A).reshape(6,-1))

# print("Latency = ")
# print(Latency(X,m,t))

# print("reward = ")
# print(reward(X,m,t, rectified_A = False, lost_penalty = 40, a = 1,l = 0, o = 0)[2])