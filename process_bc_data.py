import h5py
import numpy as np
import random

PATH_1 = "./demos/PickLargerCube-v1/motionplanning/20250308_220733.h5"
PATH_2 = "./demos/PickSmallerCube-v1/motionplanning/20250308_215535.h5"

DATASET_SIZE = 5000
TRAJ_LEN = 64

f1 = h5py.File(PATH_1, "r")

breakpoint()

f1.close()