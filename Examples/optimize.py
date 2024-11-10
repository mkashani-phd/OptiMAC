import sys
sys.path.append('../')

import src.TagModel as model
import src.TagModel_lat as model_latency
import src.Auth as Auth
import utils.utils as utils
import numpy as np
import matplotlib.pyplot as plt 
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('my_dict', type=str)
args = parser.parse_args()

parameters = json.loads(args.my_dict)


exp = utils.Run_Experiment(model        = model.math_model,
                           parameters   = parameters,
                           eval         = Auth.evaluate,
                           m_size       = 128,
                           t_size       = 256,
                           save         = True)