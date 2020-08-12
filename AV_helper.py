import argparse
import copy
import datetime
import imageio
import linecache
import math
import matplotlib as mpl
import matplotlib.cm as mcm
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import multiprocessing
import ntpath
import numpy as np
import os.path
import pandas as pd
import pickle
import platform
import psutil
import pyopencl
import random
import scipy as sp
import scipy.io as sio
import shutil
import signal
import socket
import subprocess
import sys
import time
import tracemalloc
import uuid

from collections import OrderedDict
from colorama import init, Fore, Back, Style
init()
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from math import pi
from multiprocessing import Pool, freeze_support, Process
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from PIL import Image
from scipy import fftpack, io
from skimage.transform import resize

# ROOTPATH_g = (os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")).split("/Data")[0]

# sys.path.append(os.path.normcase(ROOTPATH_g + "/Data/Work/UtilSrcCode/python-Helper"))
# from AV_helper import AV_client_id

def AV_client_id():
	# return (str(uuid.uuid1(node=None, clock_seq=None))).split("-")[-1]
	return socket.gethostname()

all_client_id_g 	= {"nam": "DESKTOP-0R5N95L", "vattha_mac": "e0f84716a2da", "yok": "DESKTOP-M6E1SUQ", "vattha": "DESKTOP-QFM8ELP", "pre": "DESKTOP-G92TU62", "thanaphon": "DESKTOP-AEALRJ5"}
all_client_name_g 	= {value:key for key, value in all_client_id_g.items()}

def AV_client_name():
	return all_client_name_g[AV_client_id()]

# Set the number of CPU for parallel processing.
client_id_l = AV_client_id()
if (client_id_l == all_client_id_g["vattha"]):
	# # N_cores_g = 1
	# N_cores_g = 11
	# # N_cores_g = 8
	# # N_cores_g = psutil.cpu_count() - 1

	pass	
elif (client_id_l == all_client_id_g["vattha_mac"]):
	# N_cores_g = 1
	pass

elif (client_id_l == all_client_id_g["pre"]):
	# # study: Mon afternoon, Tue afternoon, Wed afternoon, Fri. morning

	# # N_cores_g = 1
	# # N_cores_g = 8			# server busy load
	# # N_cores_g = 9
	# N_cores_g = 11 		# server free load
	# N_cores_g = psutil.cpu_count() - 1
	pass
elif (client_id_l == all_client_id_g["thanaphon"]):

	# # N_cores_g = 1
	# # N_cores_g = 8			# server busy load
	# N_cores_g = 11 		# server free load
	# # N_cores_g = psutil.cpu_count() - 1
	pass
elif (client_id_l == all_client_id_g["yok"]):
	# # study: Mon, Tue afternoon, Wed

	# # N_cores_g = 1
	# # N_cores_g = 4			# server busy load
	# # N_cores_g = 8
	# # N_cores_g = 10
	# N_cores_g = 11 		# server free load
	# # N_cores_g = psutil.cpu_count() - 1
	pass
elif (client_id_l == all_client_id_g["nam"]):
	# # study: Mon, Tue afternoon, Wed

	# # N_cores_g = 1
	# # N_cores_g = 4			# server busy load
	# # N_cores_g = 8
	# # N_cores_g = 10
	# N_cores_g = 11 		# server free load
	# # N_cores_g = psutil.cpu_count() - 1
	pass	
else:
	print("Unrecognized: " + client_id_l)
	# exit(0)

################################ Random ################################
def AV_rand_normal(mean_p, std_p, size_p, seed_p=None):
	if seed_p is not None:
		np.random.seed(seed_p)

	return np.random.normal(loc=mean_p, scale=std_p, size=size_p)

def AV_rand_seed(seed_p=0):
    random.seed(a=seed_p)

def AV_rand_int(N_begin_p, N_end_p, N_samples_p=None, is_allow_duplicate_p=False, seed_p=None):
	"""
	We get a N_samples_p random sequence, of which values are N_begin_p <= N <= N_end_p.
	"""
	if seed_p is not None:
		random.seed(a=seed_p)

	if N_samples_p is None:
		N_samples_l = N_end_p - N_begin_p + 1
	else:
		N_samples_l = N_samples_p

	if is_allow_duplicate_p:
		return np.asarray([random.randint(N_begin_p, N_end_p) for p in range(0, N_samples_l)], dtype=np.int32)
	else:
		return np.asarray(random.sample(range(N_begin_p, N_end_p + 1), N_samples_l), dtype=np.int32)

################################ Else ################################
def AV_hasattr(x_p, attr_p):
	return hasattr(x_p, attr_p)

def AV_tag2address(tag_p):
    if "vattha" in tag_p:
        return AV_smb_absPathToWork()[0]
          
    elif "pre" in tag_p:
        return AV_smb_absPathToWork()[1]

    elif "yok" in tag_p:
        return AV_smb_absPathToWork()[2]

    elif "thanaphon" in tag_p: 
        return AV_smb_absPathToWork()[3]

    elif "nam" in tag_p: 
        return AV_smb_absPathToWork()[4]

def AV_isclose(a_p, b_p, rtol_p=1e-05, atol_p=1e-08, equal_nan_p=False):
	"""
	Returns a boolean array where two arrays are element-wise equal within a tolerance.

	The tolerance values are positive, typically very small numbers. 
	The relative difference (rtol * abs(b)) and the absolute difference atol are added together 
	to compare against the absolute difference between a and b, i.e. absolute(a - b) <= (atol + rtol * absolute(b)).

	Parameters:	
		a, b : array_like, Input arrays to compare.
		rtol : float, The relative tolerance parameter (see Notes).
		atol : float, The absolute tolerance parameter (see Notes).
		equal_nan : bool, Whether to compare NaN’s as equal. If True, NaN’s in a will be considered equal to NaN’s in b in the output array.
	"""	

	boolean_array_l 	= np.isclose(a=a_p, b=b_p, rtol=rtol_p, atol=atol_p, equal_nan=equal_nan_p)

	idx_boolean_array_l	= np.where(boolean_array_l)

	return idx_boolean_array_l, boolean_array_l

def AV_get_all_client_id():
	return all_client_id_g

def AV_allclose(x_p, y_p, rtol_p=1e-05, atol_p=0.0, isRelToEachEle=True, isShowTypicalVal=False):
	"""
    absolute(a - b) <= (atol + rtol * absolute(b))
	"""
	if isRelToEachEle:
		return np.allclose(x_p, y_p, rtol=rtol_p, atol=atol_p)
	else:
		typical_val_l = (np.mean(np.abs(x_p)) + np.mean(np.abs(y_p)))/2.0

		if isShowTypicalVal:
			print("AV_allclose: Typical value is ", typical_val_l)

		return (not any(np.abs(x_p - y_p) > (atol_p + rtol_p * typical_val_l)))

def AV_bash_command(cmd_p):
    # subprocess.Popen(["/bin/bash", "-c", cmd_p])
    subprocess.Popen(cmd_p, shell=True, executable='/bin/bash')


def AV_power(x_p, exp_p):
# First array elements raised to powers from second array, element-wise.
	return np.float_power(x_p, exp_p)

def AV_NaNs(dims_p, dtype_p=None):
# Params:
#	dims_p: the Tuple
# Returns:
#	The numpy array

	if dtype_p is None:
		return np.full(dims_p, np.nan)
	else:
		return np.full(dims_p, np.nan, dtype=dtype_p)

def AV_Zeros(dims_p, dtype_p=None):
# Params:
#	dims_p: the Tuple
# Returns:
#	The numpy array
	
	if dtype_p is None:
		return np.full(dims_p, 0.0)
	else:
		return np.full(dims_p, 0.0, dtype=dtype_p)		

def AV_Ones(dims_p, dtype_p=None):
# Params:
#	dims_p: the Tuple
# Returns:
#	The numpy array

	if dtype_p is None:
		return np.full(dims_p, 1.0)
	else:
		return np.full(dims_p, 1.0, dtype=dtype_p)

def AV_log(x_p, base_p=10.0):
	return np.log(x_p) / np.log(base_p)

def AV_sleep(t_p=1):
# Params;
# t_p: [t_p]=seconds

	time.sleep(t_p)

def AV_last_power_of_2(x_p):
	return int(1 if (x_p == 0) else 2**math.floor(math.log2(x_p)))

def AV_next_power_of_2(x_p):
    return int(1 if (x_p == 0) else 2**math.ceil(math.log2(x_p)))

def AV_rem(a_p, b_p):
	"""
	Returns:
		It returns an integer.
	"""
	return int(a_p % b_p)

def AV_getMaxElements(data_p, isDebug_p=False):
	"""
	Return the maximum of an N-dimensional array including their indices, ignoring any NaNs. When all-NaN slices are encountered 
	a RuntimeWarning is raised and NaN is returned for that slice.
	"""

	max_val_l 			= np.nanmax(data_p)
	idx_max_val_l, _ 	= AV_isclose(a_p=data_p, b_p=AV_Ones(data_p.shape)*max_val_l)

	N_dim_l = np.ndim(data_p)

	if N_dim_l == 1:
		listOfCordinates_l = list(zip(\
			idx_max_val_l[0]))

	elif N_dim_l == 2:
		listOfCordinates_l = list(zip(\
			idx_max_val_l[0], \
			idx_max_val_l[1]))

	elif N_dim_l == 3:
		listOfCordinates_l = list(zip(\
			idx_max_val_l[0], \
			idx_max_val_l[1], \
			idx_max_val_l[2]))

	elif N_dim_l == 4:
		listOfCordinates_l = list(zip(\
			idx_max_val_l[0], \
			idx_max_val_l[1], \
			idx_max_val_l[2], \
			idx_max_val_l[3]))

	elif N_dim_l == 5:
		listOfCordinates_l = list(zip(\
			idx_max_val_l[0], \
			idx_max_val_l[1], \
			idx_max_val_l[2], \
			idx_max_val_l[3], \
			idx_max_val_l[4]))

	elif N_dim_l == 6:
		listOfCordinates_l = list(zip(\
			idx_max_val_l[0], \
			idx_max_val_l[1], \
			idx_max_val_l[2], \
			idx_max_val_l[3], \
			idx_max_val_l[4], \
			idx_max_val_l[5]))

	elif N_dim_l == 7:
		listOfCordinates_l = list(zip(\
			idx_max_val_l[0], \
			idx_max_val_l[1], \
			idx_max_val_l[2], \
			idx_max_val_l[3], \
			idx_max_val_l[4], \
			idx_max_val_l[5], \
			idx_max_val_l[6]))

	elif N_dim_l == 8:
		listOfCordinates_l = list(zip(\
			idx_max_val_l[0], \
			idx_max_val_l[1], \
			idx_max_val_l[2], \
			idx_max_val_l[3], \
			idx_max_val_l[4], \
			idx_max_val_l[5], \
			idx_max_val_l[6], \
			idx_max_val_l[7]))

	elif N_dim_l == 9:
		listOfCordinates_l = list(zip(\
			idx_max_val_l[0], \
			idx_max_val_l[1], \
			idx_max_val_l[2], \
			idx_max_val_l[3], \
			idx_max_val_l[4], \
			idx_max_val_l[5], \
			idx_max_val_l[6], \
			idx_max_val_l[7], \
			idx_max_val_l[8]))

	elif N_dim_l == 10:
		listOfCordinates_l = list(zip(\
			idx_max_val_l[0], \
			idx_max_val_l[1], \
			idx_max_val_l[2], \
			idx_max_val_l[3], \
			idx_max_val_l[4], \
			idx_max_val_l[5], \
			idx_max_val_l[6], \
			idx_max_val_l[7], \
			idx_max_val_l[8], \
			idx_max_val_l[9]))
	else:
		print("AV_getMaxElements: Not support")
		exit(0)

	return max_val_l, listOfCordinates_l 

################################ Metrics ################################
def AV_sse(a_p, b_p):
	"""
	The sum of squared error.
	"""
	return np.sum((np.array(a_p) - np.array(b_p))**2)

def AV_rmse(a_p, b_p):
	"""
	The root mean squared error.
	"""
	return np.sqrt(((a_p - b_p) ** 2).mean())

################################ Network ################################
def AV_smb_absPathToWork():

    absPathToWork_l = []

    # The 1st path is the path to the "vattha" computer.
    if AV_client_id() 	== (AV_get_all_client_id())["vattha"]:
        # absPathToWork_l.append("/media/vatthap1/Data/Work")
        absPathToWork_l.append("D:\\vattha\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["pre"]:
        absPathToWork_l.append("Z:\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["yok"]:
        absPathToWork_l.append("Z:\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["thanaphon"]:		
        absPathToWork_l.append("Z:\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["nam"]:		
        absPathToWork_l.append("Z:\\Data\\Work")

    # The 2nd path is the path to the "pre" computer.
    if AV_client_id() 	== (AV_get_all_client_id())["vattha"]:
        # absPathToWork_l.append("/run/user/1000/gvfs/smb-share:server=192.168.0.150,share=vattha/Data/Work")
        absPathToWork_l.append("F:\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["pre"]:
        absPathToWork_l.append("F:\\vattha\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["yok"]:
        absPathToWork_l.append("F:\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["thanaphon"]:
        absPathToWork_l.append("F:\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["nam"]:
        absPathToWork_l.append("F:\\Data\\Work")
     
    # The 3rd path is the path to the "yok" computer.
    if AV_client_id()   == (AV_get_all_client_id())["vattha"]:
        # absPathToWork_l.append("/run/user/1000/gvfs/smb-share:server=192.168.0.151,share=vattha/Data/Work")
        absPathToWork_l.append("G:\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["pre"]:
        absPathToWork_l.append("G:\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["yok"]:
        absPathToWork_l.append("D:\\vattha\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["thanaphon"]:       
        absPathToWork_l.append("G:\\Data\\Work")    
    elif AV_client_id() == (AV_get_all_client_id())["nam"]:       
        absPathToWork_l.append("G:\\Data\\Work")    

    # The 4th path is the path to the "thanaphon" computer.
    if AV_client_id()   == (AV_get_all_client_id())["vattha"]:
        # absPathToWork_l.append("/run/user/1000/gvfs/smb-share:server=192.168.0.152,share=vattha/Data/Work")
        absPathToWork_l.append("H:\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["pre"]:
        absPathToWork_l.append("H:\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["yok"]:
        absPathToWork_l.append("H:\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["thanaphon"]:       
        absPathToWork_l.append("E:\\vattha\\Data\\Work")    
    elif AV_client_id() == (AV_get_all_client_id())["nam"]:
        absPathToWork_l.append("I:\\Data\\Work")

    # The 5th path is the path to the "nam" computer.
    if AV_client_id()   == (AV_get_all_client_id())["vattha"]:
        # absPathToWork_l.append("/run/user/1000/gvfs/smb-share:server=192.168.0.152,share=vattha/Data/Work")
        absPathToWork_l.append("I:\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["pre"]:
        absPathToWork_l.append("I:\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["yok"]:
        absPathToWork_l.append("I:\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["thanaphon"]:       
        absPathToWork_l.append("I:\\Data\\Work")
    elif AV_client_id() == (AV_get_all_client_id())["nam"]:       
        absPathToWork_l.append("H:\\vattha\\Data\\Work")    

    return absPathToWork_l	

def AV_is_vattha():
	if AV_client_id() == (AV_get_all_client_id())["vattha"]:
		return True
	else:
		return False

def AV_is_vattha_mac():
	if AV_client_id() == (AV_get_all_client_id())["vattha_mac"]:
		return True
	else:
		return False

def AV_is_pre():
	if AV_client_id() == (AV_get_all_client_id())["pre"]:
		return True
	else:
		return False

def AV_is_yok():
	if AV_client_id() == (AV_get_all_client_id())["yok"]:
		return True
	else:
		return False		

def AV_is_thanaphon():
	if AV_client_id() == (AV_get_all_client_id())["thanaphon"]:
		return True
	else:
		return False		

def AV_is_nam():
	if AV_client_id() == (AV_get_all_client_id())["nam"]:
		return True
	else:
		return False	

################################ Memory ################################
def AV_mem_display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

################################ Array ################################
def AV_add_singleton(A_p, axis_p=-1):
	"""
	Params:
		A_p		: a numpy array
		axis_p	: at which dimension we want to add singleton, i.e. np.ndim(A_p)[axis_p] == 1
	"""

	if axis_p == -1:		
		return np.expand_dims(A_p, np.ndim(A_p))
	else:
		return np.expand_dims(A_p, axis_p)

def AV_flatten(A_p, axis_begin_p=0, axis_end_p=None):
	"""
	Params:
		A_p				: a numpy array
		axis_begin_p	: We lump the dimension from axis_begin_p to axis_end_p.
	"""

	if axis_end_p is not None:
		if A_p.ndim == (axis_end_p + 1):
			axis_end_p = None

	if axis_end_p is None:
		return A_p.reshape(*A_p.shape[:axis_begin_p], -1)
	else:
		assert axis_begin_p <= axis_end_p

		end_l = (axis_end_p + 1) - A_p.ndim
	
		return A_p.reshape(*A_p.shape[:axis_begin_p], -1, *A_p.shape[end_l:])


def AV_iscontainNaN(A_p):
	if np.isnan(A_p).any():
		return True
	else:
		return False

def AV_swapaxes(a_p, i_axis1_p, i_axis2_p):
    new_a_p = copy.deepcopy(a_p)

    return np.swapaxes(new_a_p, i_axis1_p, i_axis2_p)

def AV_axisX_concat(tuples_p, axis_p=0):
	"""
	Params:
		tuples_p	: A list of numpy arrays. Each elements of the list must have the same shape except the axis_p dimension.
	"""

	return np.concatenate(tuples_p, axis=axis_p)

def AV_axis0_concat(tuples_p):
	# c=np.vstack((a,b)), a is on top of b

	return np.vstack(tuples_p)

def AV_concat(tuples_p):
	"""
	a=[1,2], b=[3,4,5], np.hstack((a,b))==[1,2,3,4,5]
	"""
	
	return np.hstack(tuples_p)


def AV_rank1(x):
	"""
	This functions rank the items 40 times faster than AV_rank2.
	See https://stackoverflow.com/questions/36971201/map-array-of-numbers-to-rank-efficiently-in-python
	"""

	# Sort values i = 0, 1, 2, .. using x[i] as key
	y = sorted(range(len(x)), key = lambda i: x[i])

	# Map each value of x to a rank. If a value is already associated with a
	# rank, the rank is updated. Iterate in reversed order so we get the
	# smallest rank for each value.
	# rank = { x[y[i]]: i for i in xrange(len(y) -1, -1 , -1) }
	rank = { x[y[i]]: i for i in range(len(y) -1, -1 , -1) }

	# Remove gaps in the ranks
	# kv = sorted(rank.iteritems(), key = lambda p: p[1])
	kv = sorted(rank.items(), key = lambda p: p[1])

	for i in range(len(kv)):
	    kv[i] = (kv[i, 0], i)

	rank = { p[0]: p[1] for p in kv }

	# Pre allocate a array to fill with ranks
	r = np.zeros((len(x),), dtype=np.int)

	for i, v in enumerate(x):
	    r[i] = rank[v]

	return r

def AV_rank2(x):
    x_sorted = sorted(x)

    # creates a new list to preserve x
    rank = list(x)
    for v in x_sorted:
        rank[rank.index(v)] = x_sorted.index(v)

    return rank	

def AV_zeropadding(x_p, n_p):
	"""
	Padding is done by adding more columns. (np.shape(x_p)[1]) < n_p
	Params:
		x_p: row of x_p represents variables. column of x_p represents time series.	
	"""

	if (x_p.ndim == 1):
		N_times_l = len(x_p)
		N_pad_l = n_p - N_times_l
		return np.hstack((x_p, np.zeros((N_pad_l), dtype=x_p.dtype)))
	elif (x_p.ndim == 2):
		N_vars_l = np.shape(x_p)[0]
		N_times_l = np.shape(x_p)[1]

		N_pad_l = n_p - N_times_l
		return np.hstack((x_p, np.zeros((N_vars_l, N_pad_l), dtype=x_p.dtype)))

def AV_get_array_elements(array_p, N_elements_p):
	return np.take(a=array_p, indices=np.arange(N_elements_p), mode="wrap")

def AV_dsearchn(pool_p, what2look_p):
	def desearchn(pool_p, an_item_p):
	  return np.argmin(np.abs(pool_p - an_item_p))


	what2look_idx_l = []
	for an_item_l in what2look_p:
		what2look_idx_l.append(desearchn(pool_p, an_item_l))

	return np.asarray(what2look_idx_l)	

################################ Check and convert Data type ################################
def AV_is_numpy_ndarray(arr_p):
# Is it a numpy array?
	return isinstance(arr_p, np.ndarray)

def AV_is_list(arr_p):
# Is it a list?
	return isinstance(arr_p, list)

def AV_is_panda_series(arr_p):
# Is it a list?
	return isinstance(arr_p, pd.core.series.Series)

def AV_2DListToPandaDataFrame(data_p, col_labels_p=None):
	"""
	Params:
	data_p		: np.shape(data_p) == (N_observations, N_groups)
	col_labels_p: For example, col_labels_p == ['Heading1', 'Heading2'] 
	"""

	df_l = pd.DataFrame(data_p)

	if col_labels_p is not None:
		df_l.columns = col_labels_p

	return df_l


################################ Convertor ################################
def AV_interp1d(x_p, y_p, new_x_p):
	"""
	Params:
		x_p	: [time]		
	"""
	assert (x_p[0] == new_x_p[0]) and (x_p[-1] == new_x_p[-1])

	f_l = sp.interpolate.interp1d(x_p, y_p)

	return f_l(new_x_p)

def AV_print_colored_txt(txt_p, color_p="y"):

	if color_p == "m":
		header_l = Fore.MAGENTA
	elif color_p == "b":
		header_l = Fore.BLUE
	elif color_p == "c":
		header_l = Fore.CYAN
	elif color_p == "g":
		header_l = Fore.GREEN
	elif color_p == "y":
		header_l = Fore.YELLOW
	elif color_p == "r":
		header_l = Fore.RED

	return header_l + txt_p + Style.RESET_ALL

def AV_ndarray_2_list(arr_p):
# Numpy array to List

	return arr_p.tolist()

def AV_print_num(num_p, num_decimal_places_p=2):
	format_l = "%." + str(num_decimal_places_p) + "f"

	return format_l % num_p

def AV_print_object_type(obj_p):

	return type(obj_p)

def AV_print_object_attributes(obj_p, isDisplay_p=False):
	"""
	returns the __dict__ attribute for a module, class, instance, or any other object if the same has a __dict__ attribute.
	"""

	if isDisplay_p:
		for key_l, val_l in vars(obj_p).items():
			print(key_l, ":", val_l)

	return vars(obj_p)

################################ Metrics ################################
def AV_metrics_Fro(x_p):
	"""
	Frobenius norm
	"""

	return np.sqrt(np.sum(np.square(x_p)))

################################ File ################################
def AV_isDirExist(abs_dir_p, isCreate_p=True):
# This function checks if the directory (e.g. path_2_corrtime_l = path_2_dir_p + "/avgpowtime/") exists.

	try:
		if not os.path.exists(abs_dir_p):
			if isCreate_p:
				os.makedirs(abs_dir_p)
				return True
			else:
				return False
		else:
			return True
	except OSError:
		return True

def AV_isFileExist(abs_file_p):
	return os.path.isfile(abs_file_p)

def AV_delDir(abs_dir_p):
	if AV_isDirExist(abs_dir_p, isCreate_p=False):
		shutil.rmtree(abs_dir_p)

def AV_delFile(abs_file_p):
	if AV_isFileExist(abs_file_p):
		os.remove(abs_file_p)

def AV_createFile(abs_file_p, content_str_p=None):
	if (content_str_p is None):
	    with open(abs_file_p, "w+") as fp:
	        fp.write("\n")
	        fp.close()
	else:
	    with open(abs_file_p, "w+") as fp:
	        fp.write(content_str_p + "\n")
	        fp.close()		

def AV_norm_path(path_p):
	return str(Path(path_p).resolve())

def AV_get_filename(abs_path_p):
	return ntpath.basename(abs_path_p)

def AV_get_dirname(abs_path_p):
	return ntpath.dirname(abs_path_p)

def AV_recursived_dir(abs_path_p, type_p=0):
    """
    Params:
    type_p	: 0 for listing only files, 1 for only directories, 2 for both files and directories.
    """

    results_l = []
    for root_l, dirs_l, files_l in os.walk( AV_norm_path(abs_path_p) ):

        if (type_p == 0) or (type_p == 2):
            for name_l in files_l:
                results_l.append(os.path.join(root_l, name_l))

        if (type_p == 1) or (type_p == 2):
            for name_l in dirs_l:
                results_l.append(os.path.join(root_l, name_l))

    return results_l


################################ Brain constant ################################
def AV_N_freq_for_Morlet(): # Not all source codes are updated to the new function of retriving the number frequency of Morlet
	return 50

def AV_slow_drift_ubd():
	return 0.5

def AV_delta_lbd():
	return 1.5

def AV_delta_ubd():
	return 4.0

def AV_theta_lbd():
	return 4.0

def AV_theta_ubd():
	return 8.0

def AV_alpha_lbd():
	return 8.0

def AV_alpha_ubd():
	return 13.0

def AV_beta_lbd():
	return 13.0

def AV_beta_ubd():
	return 30.0

def AV_lowbeta_lbd():
	return AV_beta_lbd()

def AV_lowbeta_ubd():
	return 20.0

def AV_highbeta_lbd():
	return 20.0

def AV_highbeta_ubd():
	return AV_beta_ubd()

def AV_lower_than_delta_idx(frexlin_p):
	return (frexlin_p <= AV_delta_lbd())

def AV_delta_idx(frexlin_p):
	return (AV_delta_lbd() < frexlin_p) 	& (frexlin_p <= AV_delta_ubd())

def AV_theta_idx(frexlin_p):
	return (AV_theta_lbd() < frexlin_p) 	& (frexlin_p <= AV_theta_ubd())

def AV_alpha_idx(frexlin_p):
	return (AV_alpha_lbd() < frexlin_p) 	& (frexlin_p <= AV_alpha_ubd())

def AV_beta_idx(frexlin_p):
	return (AV_beta_lbd() < frexlin_p) 		& (frexlin_p <= AV_beta_ubd())

def AV_lowbeta_idx(frexlin_p):
	return (AV_lowbeta_lbd() < frexlin_p) 	& (frexlin_p <= AV_lowbeta_ubd())

def AV_highbeta_idx(frexlin_p):
	return (AV_highbeta_lbd() < frexlin_p) 	& (frexlin_p <= AV_highbeta_ubd())

def AV_higher_than_beta_idx(frexlin_p):
	return (AV_beta_ubd() < frexlin_p)


def getDelta(fs_p, signal_p):

    _, _, _, filtkern_hp_l, _, _, _, _, _ \
    = AV_firls_highpass_kernel(fs_p=fs_p, freq_p=AV_delta_lbd(), isShowFig_p=False, transw_p=0.89, orderlin_p=np.round(np.linspace(1100, 1100, 1)), isShowLog_p=False)

    _, _, _, filtkern_lp_l, _, _, _, _, _ \
    = AV_firls_lowpass_kernel(fs_p=fs_p, freq_p=AV_delta_ubd(), isShowFig_p=False, transw_p=0.33666666666666667, orderlin_p=np.round(np.linspace(1458, 1458, 1)), isShowLog_p=False)

    fsignal_lp_l = AV_reflected_zerophaseshift_filter(signal_p=signal_p, filtkern_b_p=filtkern_lp_l)
    fsignal_l = AV_reflected_zerophaseshift_filter(signal_p=fsignal_lp_l, filtkern_b_p=filtkern_hp_l)

    return fsignal_l 

def getTheta(fs_p, signal_p):

    _, _, _, filtkern_hp_l, _, _, _, _, _ \
    = AV_firls_highpass_kernel(fs_p=fs_p, freq_p=AV_theta_lbd(), isShowFig_p=False, transw_p=0.89, orderlin_p=np.round(np.linspace(412, 412, 1)), isShowLog_p=False)

    _, _, _, filtkern_lp_l, _, _, _, _, _ \
    = AV_fir1_lowpass_kernel(fs_p=fs_p, freq_p=AV_theta_ubd(), isShowFig_p=False, orderlin_p=np.round(np.linspace(21362, 21362, 1)), isShowLog_p=False)

    fsignal_lp_l = AV_reflected_zerophaseshift_filter(signal_p=signal_p, filtkern_b_p=filtkern_lp_l)
    fsignal_l = AV_reflected_zerophaseshift_filter(signal_p=fsignal_lp_l, filtkern_b_p=filtkern_hp_l)

    return fsignal_l


def getAlpha(fs_p, signal_p):

    _, _, _, filtkern_hp_l, _, _, _, _, _ \
    = AV_firls_highpass_kernel(fs_p=fs_p, freq_p=AV_alpha_lbd(), isShowFig_p=False, transw_p=0.2, orderlin_p=np.round(np.linspace(1095, 1095, 1)), isShowLog_p=False)

    _, _, _, filtkern_lp_l, _, _, _, _, _ \
    = AV_fir1_lowpass_kernel(fs_p=fs_p, freq_p=AV_alpha_ubd(), isShowFig_p=False, orderlin_p=np.round(np.linspace(26267, 26267, 1)), isShowLog_p=False)

    fsignal_lp_l = AV_reflected_zerophaseshift_filter(signal_p=signal_p, filtkern_b_p=filtkern_lp_l)
    fsignal_l = AV_reflected_zerophaseshift_filter(signal_p=fsignal_lp_l, filtkern_b_p=filtkern_hp_l)

    return fsignal_l


def getBeta(fs_p, signal_p):

    _, _, _, filtkern_hp_l, _, _, _, _, _ \
    = AV_firls_highpass_kernel(fs_p=fs_p, freq_p=AV_beta_lbd(), isShowFig_p=False, transw_p=0.7122222222222223, orderlin_p=np.round(np.linspace(236, 236, 1)), isShowLog_p=False)

    _, _, _, filtkern_lp_l, _, _, _, _, _ \
    = AV_fir1_lowpass_kernel(fs_p=fs_p, freq_p=AV_beta_ubd(), isShowFig_p=False, orderlin_p=np.round(np.linspace(30342, 30342, 1)), isShowLog_p=False)

    fsignal_lp_l = AV_reflected_zerophaseshift_filter(signal_p=signal_p, filtkern_b_p=filtkern_lp_l)
    fsignal_l = AV_reflected_zerophaseshift_filter(signal_p=fsignal_lp_l, filtkern_b_p=filtkern_hp_l)

    return fsignal_l    

def AV_delta_color():      
	return AV_colors("red")

def AV_theta_color():    
	return AV_colors("yellow")

def AV_alpha_color():
	return AV_colors("green")

def AV_beta_color():
	return AV_colors("violet")

def AV_lowbeta_color():
	return AV_colors("dodgerblue")

def AV_highbeta_color():
	return AV_colors("blue")

################################ Circos, MUSE ################################
def AV_circos_MUSE_TP9_delta():
	return "tp9 20 25"
def AV_circos_MUSE_TP9_theta():
	return "tp9 15 20"
def AV_circos_MUSE_TP9_alpha():
	return "tp9 10 15"
def AV_circos_MUSE_TP9_lowbeta():
	return "tp9 5 10"
def AV_circos_MUSE_TP9_highbeta():
	return "tp9 0 5"

def AV_circos_MUSE_AF7_delta():
	return "af7 0 5"
def AV_circos_MUSE_AF7_theta():
	return "af7 5 10"
def AV_circos_MUSE_AF7_alpha():
	return "af7 10 15"
def AV_circos_MUSE_AF7_lowbeta():
	return "af7 15 20"
def AV_circos_MUSE_AF7_highbeta():
	return "af7 20 25"

def AV_circos_MUSE_AF8_delta():
	return "af8 20 25"
def AV_circos_MUSE_AF8_theta():
	return "af8 15 20"
def AV_circos_MUSE_AF8_alpha():
	return "af8 10 15"
def AV_circos_MUSE_AF8_lowbeta():
	return "af8 5 10"
def AV_circos_MUSE_AF8_highbeta():
	return "af8 0 5"

def AV_circos_MUSE_TP10_delta():
	return "tp10 0 5"
def AV_circos_MUSE_TP10_theta():
	return "tp10 5 10"
def AV_circos_MUSE_TP10_alpha():
	return "tp10 10 15"
def AV_circos_MUSE_TP10_lowbeta():
	return "tp10 15 20"
def AV_circos_MUSE_TP10_highbeta():
	return "tp10 20 25"


def AV_circos_MUSE_red_lines(min_p, max_p, val_p):
# Params:
#	max_p>min_p>=0: the largest possible value

	if ((min_p < 0.0) | (max_p < 0.0) | (val_p < 0.0)):
		print("AV_circos_MUSE_red_lines: Wrong parameters")
		exit(0)

	ratio_l = (val_p - min_p)/(max_p - min_p)*100.0

	if (ratio_l < 14.29):
		return "color=reds-7-seq-1,thickness=7"
	elif (ratio_l < 28.58):
		return "color=reds-7-seq-2,thickness=11"
	elif (ratio_l < 42.87):
		return "color=reds-7-seq-3,thickness=15"
	elif (ratio_l < 57.16):
		return "color=reds-7-seq-4,thickness=19"
	elif (ratio_l < 71.45):
		return "color=reds-7-seq-5,thickness=23"
	elif (ratio_l < 85.74):
		return "color=reds-7-seq-6,thickness=27"
	else:
		return "color=reds-7-seq-7,thickness=31"


def AV_circos_MUSE_blue_lines(min_p, max_p, val_p):
# Params:
#	max_p>min_p>=0: the largest possible value

	if ((min_p < 0.0) | (max_p < 0.0) | (val_p < 0.0)):
		print("AV_circos_MUSE_blue_lines: Wrong parameters")
		exit(0)


	ratio_l = (val_p - min_p)/(max_p - min_p)*100.0

	if (ratio_l < 14.29):
		return "color=blues-7-seq-1,thickness=7"
	elif (ratio_l < 28.58):
		return "color=blues-7-seq-2,thickness=11"
	elif (ratio_l < 42.87):
		return "color=blues-7-seq-3,thickness=15"
	elif (ratio_l < 57.16):
		return "color=blues-7-seq-4,thickness=19"
	elif (ratio_l < 71.45):
		return "color=blues-7-seq-5,thickness=23"
	elif (ratio_l < 85.74):
		return "color=blues-7-seq-6,thickness=27"
	else:
		return "color=blues-7-seq-7,thickness=31"


################################ Display ################################
def AV_CMYK_to_RGB(CMYK_matrix_p):
	"""
	https://codegolf.stackexchange.com/questions/129208/convert-cmyk-values-to-rgb
	The element of CMYK_matrix_p is between 0 and 1 with the following formula

	Red   = (1 - Cyan)    x (1 - Black)   
	Green = (1 - Magenta) x (1 - Black)   
	Blue  = (1 - Yellow)  x (1 - Black)  

	Params:
		CMYK_matrix_p	: A numpy array

	Return:
		RGB_matrix_l	: Value between 0 and 1
	"""

	cyan_l 		= CMYK_matrix_p[..., 0]
	magenta_l 	= CMYK_matrix_p[..., 1]
	yellow_l 	= CMYK_matrix_p[..., 2]
	black_l 	= CMYK_matrix_p[..., 3]

	red_l 	= (1 - cyan_l)*(1 - black_l)
	green_l = (1 - magenta_l)*(1 - black_l)
	blue_l 	= (1 - yellow_l)*(1 - black_l)
	
	RGB_matrix_l = np.stack((red_l, green_l, blue_l), axis=-1)

	return RGB_matrix_l	

def AV_locations(name_p):

	# u for upper
	# l for lower and left
	# r for right
	# c for center
	locs_l = {
    'best': 0,
    'ur': 1,
    'ul': 2,
    'll': 3,
    'lr': 4,
    'r': 5,
    'cl': 6,
    'cr': 7,
    'lc': 8,
    'uc': 9,
    'c': 10}

	return locs_l[name_p]

def AV_patterns(name_p):
# hatch
	patterns_l = {
	"minus": "-",
	"plus": "+",
	"cross": "x",
	"backslash": "\\",
	"slash": "/",
	"star": "*",
	"dot": "."}

	return patterns_l[name_p]

def AV_linestyle(name_p):
	# https://matplotlib.org/gallery/lines_bars_and_markers/linestyles.html

	linestyles_l = OrderedDict(
	[('solid',               (0, ())),
	('loosely dotted',      (0, (1, 10))),
	('dotted',              (0, (1, 5))),
	('densely dotted',      (0, (1, 1))),

	('loosely dashed',      (0, (5, 10))),
	('dashed',              (0, (5, 5))),
	('densely dashed',      (0, (5, 1))),

	('loosely dashdotted',  (0, (3, 10, 1, 10))),
	('dashdotted',          (0, (3, 5, 1, 5))),
	('densely dashdotted',  (0, (3, 1, 1, 1))),

	('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
	('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
	('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

	# linestyles_l['loosely dashdotdotted']
	return linestyles_l[name_p]

def AV_colors(name_p):
	# https://matplotlib.org/examples/color/named_colors.html
	colors_l = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

	return colors_l[name_p]

def AV_markers(name_p):
	# https://matplotlib.org/api/markers_api.html
	# https://matplotlib.org/gallery/lines_bars_and_markers/marker_reference.html

	markers_l = {
	"o": "o",
	"tri_down": "v",
	"tri_up": "^",
	"tri_left": "<",
	"tri_right": ">",
	"square": "s",
	"cross": "X",
	"plus": "P",
	"diam": "D",
	"hex": "h",
	"star": "*"}

	return markers_l[name_p]

def AV_plot_color_gradients(cmap_category, cmap_list, gradient):
	# Create figure and adjust figure height to number of colormaps
	nrows = len(cmap_list)
	figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
	fig, axes = plt.subplots(nrows=nrows, figsize=(6.4, figh))
	fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)

	axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

	for ax, name in zip(axes, cmap_list):
	    ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
	    ax.text(-.01, .5, name, va='center', ha='right', fontsize=10,
	            transform=ax.transAxes)

	# Turn off *all* ticks & spines, not just the ones with colormaps.
	for ax in axes:
	    ax.set_axis_off()

def AV_plot_allcmap():
	cmaps_l = [
	('Perceptually Uniform Sequential', ['viridis', 'plasma', 'inferno', 'magma', 'cividis']),

	('Sequential', 						['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
	            						'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
	            						'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),

	('Sequential (2)', 					['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
	            						'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
	            						'hot', 'afmhot', 'gist_heat', 'copper']),

	('Diverging', 						['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
	            						'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),

	('Cyclic', 							['twilight', 'twilight_shifted', 'hsv']),

	('Qualitative', 					['Pastel1', 'Pastel2', 'Paired', 'Accent',
	            						'Dark2', 'Set1', 'Set2', 'Set3',
	            						'tab10', 'tab20', 'tab20b', 'tab20c']),

	('Miscellaneous', 					['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
	            						'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
	            						'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]

	gradient = np.linspace(0, 1, 256)
	gradient = np.vstack((gradient, gradient))

	for cmap_category, cmap_list in cmaps_l:
		AV_plot_color_gradients(cmap_category, cmap_list, gradient)

	plt.show()

def AV_twolevelsPlot(rows_1st_p, cols_1st_p, rows_2nd_p, cols_2nd_p):

	fig_l = plt.figure()

	gs_1st_l = fig_l.add_gridspec(rows_1st_p, cols_1st_p)

	gs_2nd_l = []
	for i_rows_1st_l in np.arange(rows_1st_p):
		for i_cols_1st_l in np.arange(cols_1st_p):
			gs_2nd_l = np.append(gs_2nd_l, gs_1st_l[i_rows_1st_l, i_cols_1st_l].subgridspec(rows_2nd_p, cols_2nd_p))

	# len(gs_2nd_l) == rows_1st_p*cols_1st_p. gs_2nd_l[0,..,rows_1st_p*cols_1st_p, rows_2nd_p, cols_2nd_p]
	return fig_l, gs_1st_l, gs_2nd_l

def AV_cc(arg_p, alpha_p=0.6):
    return mcolors.to_rgba(arg_p, alpha=alpha_p)

################################ Various plots ################################
def AV_plot_scatter(\
    x_p, y_p, \
    x_label_p, y_label_p, \
    xlim_min_p=None, xlim_max_p=None, \
    ylim_min_p=None, ylim_max_p=None, \
    x_hist_bins_p=100, y_hist_bins_p=100):

    fig = plt.figure(figsize=(8,8))
    gs_l = gridspec.GridSpec(3, 3)

    # The scatter plot
    ax_main_l   = plt.subplot(gs_l[1:3, :2])
    
    ax_main_l.scatter(x_p, y_p, marker=".")
    ax_main_l.set(xlabel=x_label_p, ylabel=y_label_p)

    if xlim_min_p is not None:
        plt.xlim([xlim_min_p - 0.05, xlim_max_p + 0.05])

    if ylim_min_p is not None:
        plt.ylim([ylim_min_p - 0.05, ylim_max_p + 0.05])

    # plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)


    # The histogram of the X data
    ax_xDist_l  = plt.subplot(gs_l[0, :2], sharex=ax_main_l)

    ax_xDist_l.hist(x_p, bins=x_hist_bins_p, align="mid")
    ax_xDist_l.set(ylabel="count")
    # ax_xCumDist = ax_xDist.twinx()
    # ax_xCumDist.hist(x_p, bins=x_hist_bins_p, cumulative=True, histtype="step", normed=True, color='r',align='mid')
    # ax_xCumDist.tick_params('y', colors='r')
    # ax_xCumDist.set_ylabel('cumulative',color='r')

    # The histogram of the Y data
    ax_yDist_l  = plt.subplot(gs_l[1:3, 2], sharey=ax_main_l)

    ax_yDist_l.hist(y_p, bins=y_hist_bins_p, orientation="horizontal", align="mid")
    ax_yDist_l.set(xlabel="count")
    # ax_yCumDist = ax_yDist.twiny()
    # ax_yCumDist.hist(y,bins=100,cumulative=True,histtype='step',normed=True,color='r',align='mid',orientation='horizontal')
    # ax_yCumDist.tick_params('x', colors='r')
    # ax_yCumDist.set_xlabel('cumulative',color='r')


    plt.show()


def AV_plot_waterfall(axe_p, \
	x_p, y_p, C_p, \
	x_colors_lin_p, \
	C_min_p, C_max_p, x_colors_alpha_p=0.6):
	"""
	This graph is suitable when N << M.

	Params:
		x_p : len(x_p)=N
		y_p : len(y_p)=M
		C_p : C_p[i_y,i_x] is mapped to x_p[i_x] and y_p[i_y]. That is the row of C_p is for y_p and the column is for x_p.
		x_colors_lin_p : len(x_colors_lin_p)=N. It is a Lists datatype of color string, e.g. "r", "g"
	"""

	# Rename the colors.
	facecolors_l = []
	for x_colors_l in x_colors_lin_p:
		facecolors_l.append(AV_cc(arg_p=x_colors_l, alpha_p=x_colors_alpha_p))

	verts_l = []
	for i_x_l in range(len(x_p)):        
		verts_l.append(list(zip(y_p, C_p[:, i_x_l])))

	poly_l =  PolyCollection(verts_l, facecolors=facecolors_l)
	# poly_l.set_alpha(0.7)
	axe_p.add_collection3d(poly_l, zs=x_p, zdir="y")

	axe_p.set_xlabel('X')
	axe_p.set_xlim3d(0, len(y_p))
	axe_p.set_ylabel('Y')
	axe_p.set_ylim3d(0, len(x_p))
	axe_p.set_zlabel('Z')
	axe_p.set_zlim3d(C_min_p, C_max_p)


def AV_radar_chart_MUSE(ax_p, chns_label_p, data_p=None, N_circles_p=3, max_radial_p=None, alpha_p=0.1, linewidth_p=1, isShowLegend_p=True, marker_p="o"):
	"""
	ax_p:			This ax_p is generated by "plt.subplot(111, polar=True)".
	chns_label_p:	List of string chns_label_p[0] <=> data_p[0, :].
	N_circles_p:	The number of circles excluding the outermost circle and the point at the center.
	data_p: 		List of (N_chns_l, N_bands_l). This function is created for MUSE. Therefore, N_chns_l==4
					but N_bands_l can vary.
	"""

	if data_p is None:
		data_p = AV_Ones((4, 6))*10

	N_chns_l 	= np.shape(data_p)[0]
	N_bands_l 	= np.shape(data_p)[1]

	polar_stepsize_l = (2.0*np.pi)/N_chns_l
	angles_l = [0.0*polar_stepsize_l, 1.0*polar_stepsize_l, 2.0*polar_stepsize_l, 3.0*polar_stepsize_l, 0.0*polar_stepsize_l]


	# print(angles_l)

	# import pandas as pd
	# from math import pi

	# ------- PART 1: Create background
	# If you want the first axis to be on top:
	ax_p.set_theta_offset(-pi / 4 - pi / 2)
	ax_p.set_theta_direction(-1)
	 
	# Draw one axe per variable + add labels labels yet
	plt.xticks(angles_l[:-1], chns_label_p)
	
	if max_radial_p is None:
		max_val_l = np.max(data_p)
	else:
		max_val_l = max_radial_p

	yticks_stepsize_l = max_val_l/(N_circles_p + 1)
	yticks_l = np.arange(N_circles_p + 1)*yticks_stepsize_l

	# Draw ylabels
	ax_p.set_rlabel_position(0)
	plt.yticks(yticks_l, [], color="grey", size=7)
	plt.ylim(0, max_val_l)
	 
	 
	# ------- PART 2: Add plots	 
	id_band_l = -1

	# Delta
	id_band_l	= id_band_l + 1
	values_l	= data_p[:, id_band_l]	
	values_l	= np.append(values_l, values_l[0])
	# ax_p.plot(angles_l, values_l, color=AV_delta_color(), linewidth=linewidth_p, linestyle='solid', label="Delta band")
	ax_p.plot(angles_l, values_l, marker_p, color=AV_delta_color(), label="Delta band")
	# yerr_l = [0.05, 0.0875, 0.125, 0.1625, 0.2]
	# yerr_l = [0.05, 0.05, 0.05, 0.05, 0.05]

	# # yerr_l = np.linspace(0.05, 0.2, 5)

	# # ax_p.errorbar(angles_l, values_l, color=AV_delta_color(), linewidth=linewidth_p, linestyle='solid', label="Delta band", yerr = np.linspace(0.05, 0.2, 5), capsize=0)
	# ax_p.errorbar(angles_l, values_l, color=AV_delta_color(), linewidth=linewidth_p, linestyle='solid', label="Delta band", yerr = 0.1, capsize=2)

	ax_p.fill(angles_l, values_l, AV_delta_color(), alpha=alpha_p)

	# Theta
	id_band_l = id_band_l + 1
	values_l	= data_p[:, id_band_l]	
	values_l	= np.append(values_l, values_l[0])
	ax_p.plot(angles_l, values_l, marker_p, color=AV_theta_color(), label="Theta band")
	ax_p.fill(angles_l, values_l, AV_theta_color(), alpha=alpha_p)

	# Alpha
	id_band_l = id_band_l + 1
	values_l	= data_p[:, id_band_l]	
	values_l	= np.append(values_l, values_l[0])
	ax_p.plot(angles_l, values_l, marker_p, color=AV_alpha_color(), label="Alpha band")
	ax_p.fill(angles_l, values_l, AV_alpha_color(), alpha=alpha_p)

	# Beta
	id_band_l = id_band_l + 1
	values_l	= data_p[:, id_band_l]	
	values_l	= np.append(values_l, values_l[0])
	ax_p.plot(angles_l, values_l, marker_p, color=AV_beta_color(), label="Beta band")
	ax_p.fill(angles_l, values_l, AV_beta_color(), alpha=alpha_p)

	# Low beta
	id_band_l = id_band_l + 1
	values_l	= data_p[:, id_band_l]	
	values_l	= np.append(values_l, values_l[0])
	ax_p.plot(angles_l, values_l, marker_p, color=AV_lowbeta_color(), label="Low beta band")
	ax_p.fill(angles_l, values_l, AV_lowbeta_color(), alpha=alpha_p)

	# High beta
	id_band_l = id_band_l + 1
	values_l	= data_p[:, id_band_l]	
	values_l	= np.append(values_l, values_l[0])
	ax_p.plot(angles_l, values_l, marker_p, color=AV_highbeta_color(), label="High beta band")
	ax_p.fill(angles_l, values_l, AV_highbeta_color(), alpha=alpha_p)

	if isShowLegend_p:
		# Add legend
		# plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
		# plt.legend(loc="best", bbox_to_anchor=(0.1, 0.1))
		pass
		

def AV_polarError(ax, r, dr, theta, numpoints = 15, color = (0,0,0), dtheta = 0.1, lw=0.7):
    for j in range( theta.shape[0] ):

        for p in [-1, +1]:
            local_theta = np.linspace(-dtheta, dtheta, 15) + theta[j]
            local_r = np.ones(15) * ( r[j] + p*dr[j] )
            ax.plot(local_theta, local_r, color=color, lw=lw, marker='', zorder=-1)

            local_theta = [theta[j]]*2
            local_r     = [ r[j], (r[j] + p*dr[j])*0.999 ]
            ax.plot(local_theta, local_r, color=color, lw=lw, marker='', zorder=-1)

    return   

def AV_plot_multiset_stackedbar(axe_p, \
		data_p, \
		# Params for subjects
		labels_subjects_p, \
		labels_subjects_size_p, \
		# Params for conditions
		colors_conds_p, \
		labels_conds_p, \
		labels_conds_size_p, \
		# Params for stacks
		patterns_stacks_p, \
		labels_stacks_p, \
		yerr_data_p=None, \
		isShow_labels_subjects_p=False, \
		isShow_labels_conds_p=True, \
		space_btw_subjects=0.50, \
		rotation_p=45, \
		legend_elements_size_p=40, \
		isShow_labels_conds_onXaxis_p=True, \
		# Params for subjects
		is_labels_subjects_onTopXaxis_p=True):
# Params:
#	axe_p:
# 	data_p: the numpy array with three elements (N_subjects_l, N_conds_l, N_stacks_l)
# 	yerr_data_p: the numpy array with three elements (N_subjects_l, N_conds_l, N_stacks_l). See https://matplotlib.org/gallery/lines_bars_and_markers/bar_stacked.html#sphx-glr-gallery-lines-bars-and-markers-bar-stacked-py
#	labels_subjects_p: the List with N_subjects_l elements
#	colors_conds_p: the List with N_conds_l elements
#	labels_conds_p: the List with N_conds_l elements
# 	patterns_stacks_p: the List
#	labels_stacks_p: the List with N_stacks_l elements


	if (int(np.shape(np.shape(data_p))[0]) == 1):
		tmp_data_p = AV_NaNs((np.shape(data_p)[0], 1, 1))
		tmp_data_p[:, 0, 0] = data_p
		data_p = tmp_data_p

	N_subjects_l 	= np.shape(data_p)[0]
	N_conds_l 		= np.shape(data_p)[1]
	N_stacks_l		= np.shape(data_p)[2]	# When there is no stack, N_stacks_l == 1

	# set width of bar
	barWidth_l 		= (1.0 - space_btw_subjects)/float(N_conds_l)
	halfwidth_l 	= np.ceil(N_conds_l/2)*barWidth_l

	width_each_subject = barWidth_l*N_conds_l

	xtick_l = []
	xtick_label_l = []

	r_l = 1.0 + np.arange(N_subjects_l) - (1.0 - space_btw_subjects)/2.0 + barWidth_l/2.0
	for idx_conds in range(N_conds_l):		

		bottom_l = AV_Zeros(N_subjects_l)
		for idx_stacks in range(N_stacks_l):
			data_l = data_p[:, idx_conds, idx_stacks]

			if (patterns_stacks_p is not None):
				if (yerr_data_p is None):
					axe_p.bar(x=r_l, height=data_l, width=barWidth_l, bottom=bottom_l, color=colors_conds_p[idx_conds], edgecolor='white', hatch=patterns_stacks_p[idx_stacks])
				else:
					yerr_data_l = yerr_data_p[:, idx_conds, idx_stacks]
					axe_p.bar(x=r_l, height=data_l, yerr=yerr_data_l, capsize=7, width=barWidth_l, bottom=bottom_l, color=colors_conds_p[idx_conds], edgecolor='white', hatch=patterns_stacks_p[idx_stacks])
			else:
				if (yerr_data_p is None):
					axe_p.bar(x=r_l, height=data_l, width=barWidth_l, bottom=bottom_l, color=colors_conds_p[idx_conds], edgecolor='white')
				else:
					yerr_data_l = yerr_data_p[:, idx_conds, idx_stacks]
					axe_p.bar(x=r_l, height=data_l, yerr=yerr_data_l, capsize=7, width=barWidth_l, bottom=bottom_l, color=colors_conds_p[idx_conds], edgecolor='white')

			bottom_l = bottom_l + data_l

		if isShow_labels_conds_p and isShow_labels_conds_onXaxis_p:
			xtick_l = np.append(xtick_l, r_l)
			xtick_label_l = np.append(xtick_label_l, np.full((N_subjects_l), labels_conds_p[idx_conds]))

		r_l = r_l + barWidth_l

	if isShow_labels_subjects_p and (not is_labels_subjects_onTopXaxis_p):
		r_l = 1.0 + np.arange(N_subjects_l)

		xtick_l = np.append(r_l, xtick_l)
		xtick_label_l = np.append(labels_subjects_p, xtick_label_l)

	AV_ax_xtick(axe_p=axe_p, xtick_p=xtick_l)
	AV_ax_xticklabel(axe_p=axe_p, xtick_label_p=xtick_label_l)

	if (patterns_stacks_p is not None):
		legend_elements_l = []

		for i_l in range(N_stacks_l):
			e_l = AV_ax_legend_elements(type_p=2, label_p=labels_stacks_p[i_l], color_p=AV_colors("white"), pattern_p=patterns_stacks_p[i_l])
			legend_elements_l.append(e_l)

		y="Adf"
		x = u"$" + y +"$"
		# # # # blue_star = Line2D([], [], color='blue', marker='*', linestyle='None', markersize=10, label='Blue stars')
		# # # # blue_star = Line2D([], [], color='blue', marker=u'$\u22a2\!\u22a3\u0041$', linestyle='None', markersize=100, label='Blue stars')
		# e_l = Line2D([], [], color='blue', marker=x, linestyle='None', markersize=100, label='Blue stars')
		
		# e_l = AV_ax_legend_elements(type_p=3, marker_str_p="DCh", markersize_p=50, label_p="DCh", color_p=AV_colors("black"))

		# legend_elements_l.append(e_l)

		AV_ax_legend(axe_p=axe_p, legend_elements_p=legend_elements_l, size_p=legend_elements_size_p)



	if not isShow_labels_conds_onXaxis_p:

		legend_elements_l = []
		for i_l in range(len(colors_conds_p)):
			e_l = AV_ax_legend_elements(type_p=0, label_p=labels_conds_p[i_l], color_p=AV_colors(colors_conds_p[i_l]))
			legend_elements_l.append(e_l)

		AV_ax_legend(axe_p=axe_p, legend_elements_p=legend_elements_l, size_p=legend_elements_size_p)		

	xtick_l = axe_p.xaxis.get_major_ticks()
	N_xtick_l = len(xtick_l)

	cnt_l = -1
	if isShow_labels_subjects_p and (not is_labels_subjects_onTopXaxis_p):	
		for i_l in range(N_subjects_l):
			cnt_l = cnt_l + 1					
			xtick_l[i_l].label.set_fontsize(labels_subjects_size_p)

	if isShow_labels_conds_p:
		for i_l in np.arange(cnt_l + 1, N_xtick_l):
			xtick_l[i_l].label.set_fontsize(labels_conds_size_p)

		AV_ax_xticklabel_rotation(axe_p=axe_p, rotation_p=rotation_p)		

	axe_p.set_xlim(1 - halfwidth_l - barWidth_l, N_subjects_l + barWidth_l + halfwidth_l)

	if is_labels_subjects_onTopXaxis_p:
		axe2_l = AV_ax_xtick_xticklabel_top(axe_p=axe_p, xtick_p=1.0 + np.arange(N_subjects_l), xtick_label_p=labels_subjects_p, minor_p=False)
		axe2_l.set_xlim(1 - halfwidth_l - barWidth_l, N_subjects_l + barWidth_l + halfwidth_l)

def AV_plot_heatmapV2(axe_p, C_p, 
    x_p=None, y_p=None, \
    cmap_p="jet", \
    shading_p="flat", \
    xscale_p="linear", \
    yscale_p="linear", \
    is_sym_cmap_p=False, \
    clamped_C_min_p=None, clamped_C_max_p=None):

    if x_p is None:
        x_l = np.arange(np.shape(C_p)[1])
    else:
        x_l = x_p

    if y_p is None:
        y_l = np.arange(np.shape(C_p)[0])
    else:
        y_l = y_p


    return AV_plot_heatmap(axe_p=axe_p, x_p=x_l, y_p=y_l, C_p=C_p, \
    	cmap_p=cmap_p, \
    	shading_p=shading_p, \
        xscale_p=xscale_p, \
        yscale_p=yscale_p, \
        is_sym_cmap_p=is_sym_cmap_p, \
        clamped_C_min_p=clamped_C_min_p, clamped_C_max_p=clamped_C_max_p)

def AV_plot_heatmap(axe_p, x_p, y_p, C_p, \
	edgecolors_p="none", cmap_p="jet", \
	shading_p="flat", \
	xscale_p="linear", xscale_base_p=10.0, xminor_p=False, \
	yscale_p="linear", yscale_base_p=10.0, yminor_p=False, \
	is_sym_cmap_p=False, \
	clamped_C_min_p=None, clamped_C_max_p=None):
# Params:
#	x_p : len(x_p)=N
#	y_p : len(y_p)=M
#	C_p : C_p[i_y,i_x] is mapped to x_p[i_x] and y_p[i_y]. That is the row of C_p is for y_p and the column is for x_p. 
#	shading_p : "flat" or "gouraud"
#	clamped_C_min_p : The patch corresponding to C_p[i_y,i_x] <= clamped_C_min_p has the lowest color.
#	clamped_C_max_p : The patch corresponding to C_p[i_y,i_x] >= clamped_C_max_p has the highest color.

	if (clamped_C_min_p is None):
		vmin_l = np.nanmin(C_p)
	else:
		vmin_l = clamped_C_min_p

	if (clamped_C_max_p is None):
		vmax_l = np.nanmax(C_p)
	else:
		vmax_l = clamped_C_max_p

	## X-axis
	if (xscale_p == "linear"):
		if (shading_p == "flat"):
			dx_l = x_p[1] - x_p[0]
			x_p = x_p - dx_l/2.0
			x_p = np.append(x_p, x_p[-1] + dx_l)
		
		axe_p.set_xscale(xscale_p)
	elif (xscale_p == "log"):
		if (shading_p == "flat"):
			x_exponent_l = AV_log(x_p=x_p, base_p=xscale_base_p)
			x_exponent_l = x_exponent_l - 0.5
			x_exponent_l = np.append(x_exponent_l, x_exponent_l[-1] + 1.0)
			x_p = xscale_base_p**x_exponent_l
		
		AV_ax_xscale(axe_p=axe_p, xscale_p=xscale_p, xscale_base_p=xscale_base_p)
		majtick_l, mintick_l = AV_ax_make_tick(var_p=x_p, N_minor_tick_l=8, varscale_p=xscale_p, varscale_base_p=xscale_base_p)
		AV_ax_xtick(axe_p=axe_p, xtick_p=majtick_l, minor_p=xminor_p)
		AV_ax_xtick(axe_p=axe_p, xtick_p=mintick_l, minor_p=True)		

	## Y-axis
	if (yscale_p == "linear"):
		if (shading_p == "flat"):
			dy_l = y_p[1] - y_p[0]
			y_p = y_p - dy_l/2.0			
			y_p = np.append(y_p, y_p[-1] + dy_l)
		
		axe_p.set_yscale(yscale_p)
	elif (yscale_p == "log"):
		# if (shading_p == "flat"):
		# 	y_exponent_l = AV_log(x_p=y_p, base_p=yscale_base_p)
		# 	# y_exponent_l = y_exponent_l - 0.5
		# 	# y_exponent_l = np.append(y_exponent_l, y_exponent_l[-1] + 1.0)
		# 	y_p = yscale_base_p**y_exponent_l

		AV_ax_yscale(axe_p=axe_p, yscale_p=yscale_p, yscale_base_p=yscale_base_p)
		# majtick_l, mintick_l = AV_ax_make_tick(var_p=y_p, N_minor_tick_l=3, varscale_p=yscale_p, varscale_base_p=yscale_base_p)
		majtick_l, mintick_l = AV_ax_make_tick(var_p=y_p, N_minor_tick_l=10, varscale_p=yscale_p, varscale_base_p=yscale_base_p)
		AV_ax_ytick(axe_p=axe_p, ytick_p=majtick_l, minor_p=yminor_p)
		AV_ax_ytick(axe_p=axe_p, ytick_p=mintick_l, minor_p=True)	

	AV_ax_tickwidth(axe_p=axe_p)

	if is_sym_cmap_p:
		max_mag_l = np.max([np.abs(vmin_l), np.abs(vmax_l)])
		vmax_l = max_mag_l
		vmin_l = -max_mag_l

	X_l, Y_l = np.meshgrid(x_p, y_p)

	cmap_l = mcm.get_cmap(cmap_p)
	# cmap_l.set_bad(color = "k", alpha = 1.)

	c_l = axe_p.pcolormesh(X_l, Y_l, C_p, edgecolors=edgecolors_p, cmap=cmap_l, vmin=vmin_l, vmax=vmax_l, shading=shading_p)

	return c_l

def AV_plot_linesegments(axe_p, x_p, y_p, color_codes_p, N_segments_p, linewidths_p, linestyles_p, color_ontop_codes_p):
# Params:
#   color: one of {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}
#   linestyles_p: Either one of [ 'solid' | 'dashed' | 'dashdot' | 'dotted' ]

    N_y_l = np.shape(y_p)[0]
    # print(N_y_l)
    N_points_in_one_segment_l = int(np.ceil(N_y_l/N_segments_p))
    N_colors_in_one_segment_l = np.shape(color_codes_p)[0]
    N_points_in_one_color_l = int(np.ceil(N_points_in_one_segment_l/N_colors_in_one_segment_l))
    # print(N_points_in_one_color_l)

    N_ontop_lines_l = np.shape(color_ontop_codes_p)[0]
    idx_colors_in_segments_l = -1
    for idx_segments_l in np.arange(N_segments_p):
        for idx_colors_in_one_segment_l in np.arange(N_colors_in_one_segment_l):
            idx_colors_in_segments_l = idx_colors_in_segments_l + 1

            idx_start_l = N_points_in_one_color_l*idx_colors_in_segments_l
            idx_stop_l = idx_start_l + N_points_in_one_color_l + 1

            if (N_y_l <= idx_stop_l):
                idx_stop_l = N_y_l - 1

            # print(idx_start_l, ",", idx_stop_l)
            # print(y_p[idx_start_l:idx_stop_l])
            # x_p[idx_start_l:idx_stop_l]

            xs_l = x_p[idx_start_l:idx_stop_l]
            # print(np.shape(xs_l))
            ys_l = [y_p[idx_start_l:idx_stop_l]]
            # tmp_x_l = x_p[]
            # tmp_y_l = y_p

            if (N_ontop_lines_l > 1):
	            # Plot backgrond lines
	            for lw_l in np.arange(N_ontop_lines_l, 0, -1):
	                linewidths_l = 3*lw_l*linewidths_p
	                # print(linewidths_l)

	                # ax.plot(x_p, y_p, linewidth=linewidths_l, color=color_ontop_codes_p[lw_l - 1])
	                line_segments_l = LineCollection([np.column_stack([xs_l, y]) for y in ys_l], linewidths=linewidths_l, colors=color_ontop_codes_p[lw_l - 1], linestyles="solid")
	                # line_segments_l.set_edgecolor("w")
	                axe_p.add_collection(line_segments_l)

            line_segments_l = LineCollection([np.column_stack([xs_l, y]) for y in ys_l], linewidths=linewidths_p, colors=color_codes_p[idx_colors_in_one_segment_l], linestyles=linestyles_p)
            # line_segments_l.set_edgecolor("w")
            axe_p.add_collection(line_segments_l)

    # ax.set_xlim(0, 101)
    # ax.set_ylim(0, 101)
    # plt.show()            
    # # for lw_l in np.arange(N_ontop_lines_l, 0, -1):
    # #     linewidths_l = lw_l*linewidths_p
    # #     # print(linewidths_l)
    # # # line_segments = LineCollection([np.column_stack([x, y]) for y in ys], linewidths=linewidths_p, linestyles=linestyles_p)
    # # # line_segments.set_array(x)

########################################### Image ###########################################
def AV_read_image(abs_path_p, scale_w_p=0, scale_h_p=0):

    img_l = Image.open(abs_path_p)

    if np.shape(img_l)[2] == 4:
        img_l = img_l.convert("RGB")

    img_l = np.array(img_l)

    if (scale_w_p*scale_h_p):
        img_l = resize(img_l, (scale_w_p, scale_h_p))*255.0

    img_l = img_l.astype("uint8")
    print("img min=", np.min(img_l), "/img max=",np.max(img_l), "/shape=", img_l.shape)

    return img_l

    # if image.mode == 'RGB':
    #     cmyk_image = image.convert('CMYK')

    # exit(code=0)
    # img_l = imageio.imread(abs_path_p).astype(np.float32)

    # if img_l.shape[2] == 4:
    # ## Convert CMYK to RGB
    #     print(np.max(img_l))
    #     rgb_scale_l = 255.0
    #     cmyk_scale_l = 100.0

    #     c_l = img_l[:, :, 0]
    #     m_l = img_l[:, :, 1]
    #     y_l = img_l[:, :, 2]
    #     k_l = img_l[:, :, 3]
    
    #     tmp_img_l = AV_Ones( (img_l.shape[0], img_l.shape[1], 3) )
    #     tmp_img_l[:, :, 0] = rgb_scale_l * (1.0 - c_l / float(cmyk_scale_l)) * (1.0 - k_l / float(cmyk_scale_l))
    #     tmp_img_l[:, :, 1] = rgb_scale_l * (1.0 - m_l / float(cmyk_scale_l)) * (1.0 - k_l / float(cmyk_scale_l))
    #     tmp_img_l[:, :, 2] = rgb_scale_l * (1.0 - y_l / float(cmyk_scale_l)) * (1.0 - k_l / float(cmyk_scale_l))

    #     img_l = tmp_img_l

    # if (scale_w_p*scale_h_p):
    #     img_l = resize(img_l, (scale_w_p, scale_h_p))

    # return img_l.astype("uint8") # Convert to the range [0, 255]

########################################### Fig's properties ###########################################
def AV_show_plot(fig_p=None):
	fig = plt.figure(num=fig_p.number)

	plt.show(block=False)
	plt.pause(1.0)
	fig.canvas.manager.window.activateWindow()
	fig.canvas.manager.window.raise_()

def AV_fig_title(fig_p, title_p, title_size_p=25):
	fig_p.suptitle(title_p, fontsize=title_size_p)

def AV_fig_xlabel(fig_p, xlabel_p, xlabel_size_p=20):
	fig_p.text(0.5, 0.04, xlabel_p, ha='center', fontsize=xlabel_size_p)

def AV_fig_ylabel(fig_p, ylabel_p, ylabel_size_p=20):
	fig_p.text(0.04, 0.5, ylabel_p, va='center', rotation='vertical', fontsize=ylabel_size_p)

def AV_fig_maximized(fig_p):
	plt.figure(num=fig_p.number)
	plt.pause(1.0)
	graphic_backend_l = plt.get_backend().lower()
	if ((graphic_backend_l == "qt4agg") | (graphic_backend_l == "qt5agg")):
		mng_l = plt.get_current_fig_manager()  # Open directly in full window

		mng_l.window.showMaximized()
	elif (graphic_backend_l == "wxagg"):
		mng_l = plt.get_current_fig_manager()  # Open directly in full window

		mng_l.frame.Maximize(True)
	elif (graphic_backend_l == "tkagg"):
		mng_l = plt.get_current_fig_manager()  # Open directly in full window

		mng_l.resize(*mng_l.window.maxsize())
	
	plt.show(block=False)
	plt.pause(1.0)

def AV_fig_savefig(fig_p, abs_path_FN_p):
	print("AV_fig_savefig: " + abs_path_FN_p)
	fig_p.savefig(abs_path_FN_p, bbox_inches = 'tight', pad_inches = 0)

def AV_fig_closefig(fig_p):
	plt.close(fig_p)

########################### XY-axis's properties ###########################
def AV_ax_get_fig(axe_p):
	return axe_p.get_figure()

def AV_ax_title(axe_p, title_p, title_size_p=20):
	axe_p.set_title(title_p, fontsize=title_size_p)

def AV_ax_make_tick(var_p, \
	N_minor_tick_l=5, \
	varscale_p="linear", varscale_base_p=10.0):

	if (varscale_p == "linear"):
		pass
	else:
		x_exponent_l 			= AV_log(x_p=var_p, base_p=varscale_base_p)
		x_exponent_min_l 		= int(np.ceil(np.min(x_exponent_l)))
		x_exponent_truemin_l 	= int(np.floor(np.min(x_exponent_l)))
		x_exponent_max_l 		= int(np.floor(np.max(x_exponent_l)))
		x_exponent_truemax_l 	= int(np.ceil(np.max(x_exponent_l)))

		majtick_l = AV_power(varscale_base_p, np.arange(x_exponent_min_l, x_exponent_max_l + 1))

		mintick_template_l = np.linspace(1.0, varscale_base_p, N_minor_tick_l + 2)

		mintick_l = []
		for i_l in np.arange(x_exponent_truemin_l, x_exponent_truemax_l + 1, 1):			
			tmp_l = mintick_template_l[1:(N_minor_tick_l + 1)]*AV_power(varscale_base_p, i_l)
			mintick_l = np.append(mintick_l, tmp_l)

		return majtick_l, mintick_l

def AV_ax_tickwidth(axe_p, tickwidth_p=2, ticklength_p=10):
	axe_p.tick_params(axis="both", which="major", width=tickwidth_p, length=ticklength_p)
	axe_p.tick_params(axis="both", which="minor", width=tickwidth_p, length=ticklength_p/2.0)

def AV_ax_legend(axe_p, legend_elements_p, location_p=AV_locations("best"), size_p=24):
	leg_l = axe_p.legend(handles=legend_elements_p, loc=location_p, prop={"size": size_p})

def AV_ax_legend_elements(type_p, label_p, \
	color_p=AV_colors("blue"), \
	# Line element
	lw_p=4, \
	# Marker element
	marker_p=AV_markers("o"), markersize_p=15, \
	# String Marker element
	marker_str_p="o", \
	# Patch element
	pattern_p=None, edgecolor_p=AV_colors("black")):

	if (type_p == 0):
		# Line element
		e_l = Line2D([0], [0], color=color_p, lw=lw_p, label=label_p)
	elif (type_p == 1):
		# Marker element
		e_l = Line2D([0], [0], marker=marker_p, color=AV_colors("white"), label=label_p, markerfacecolor=color_p, markersize=markersize_p),
	elif (type_p == 2):
		# Patch element
		e_l = Patch(facecolor=color_p, edgecolor=edgecolor_p, label=label_p, hatch=pattern_p)
	elif (type_p == 3):
		# String Marker element
		x_l = u"$" + marker_str_p +"$"
		e_l = Line2D([], [], color=color_p, marker=x_l, linestyle='None', markersize=markersize_p, label=label_p)

	return e_l

########################### X-axis's properties ###########################
def AV_ax_xlabel(axe_p, xlabel_p, xlabel_size_p=20):
	axe_p.set_xlabel(xlabel_p, fontsize=xlabel_size_p)

def AV_ax_xlabel_fontsizel(axe_p, xlabel_size_p=20):
	axe_p.set_xlabel(axe_p.get_xlabel(), fontsize=xlabel_size_p)

def AV_ax_xscale(axe_p, xscale_p, xscale_base_p=10):
	if (xscale_p == "linear"):
		axe_p.set_xscale(xscale_p)
	elif (xscale_p == "log"):
		axe_p.set_xscale(xscale_p, basex=xscale_base_p)

def AV_ax_xtick(axe_p, xtick_p, minor_p=False):
	if (type(xtick_p).__name__ == "list") & (np.shape(xtick_p)[0] == 0):
		# This is true when ytick_p = []
		axe_p.xaxis.set_major_locator(plt.NullLocator())
		axe_p.xaxis.set_minor_formatter(plt.NullFormatter())
	else:
		axe_p.set_xticks(xtick_p, minor=minor_p)
		axe_p.xaxis.set_minor_formatter(plt.NullFormatter())

def AV_ax_xtick_xticklabel_top(axe_p, xtick_p, xtick_label_p, minor_p):
	axe_l = axe_p.twiny()
	AV_ax_xtick(axe_p=axe_l, xtick_p=xtick_p, minor_p=minor_p)
	AV_ax_xticklabel(axe_p=axe_l, xtick_label_p=xtick_label_p)

	return axe_l

# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import host_subplot
# import mpl_toolkits.axisartist as AA
# import numpy as np

# ax = host_subplot(111)
# xx = np.arange(0, 2*np.pi, 0.01)
# ax.plot(xx, np.sin(xx))

# ax2 = ax.twin()  # ax2 is responsible for "top" axis and "right" axis
# ax2.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
# ax2.set_xticklabels(["$0$", r"$\frac{1}{2}\pi$",
#                      r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])

# ax2.axis["right"].major_ticklabels.set_visible(False)
# ax2.axis["top"].major_ticklabels.set_visible(True)

# plt.draw()
# plt.show()


def AV_ax_xticklabel(axe_p, xtick_label_p):
	if (type(xtick_label_p).__name__ == "list") & (np.shape(xtick_label_p)[0] == 0):
		# This is true when xtick_label_p = []
		axe_p.xaxis.set_major_formatter(plt.NullFormatter())
	else:
		axe_p.set_xticklabels(xtick_label_p)

def AV_ax_xticklabel_rotation(axe_p, rotation_p):
	axe_p.set_xticklabels(axe_p.get_xticklabels(), rotation=rotation_p, ha="right")

def AV_ax_xticklabel_fontsize(axe_p, xtick_label_fontsize_p=15):
	for tick_l in axe_p.xaxis.get_major_ticks():
		tick_l.label.set_fontsize(xtick_label_fontsize_p)		

def AV_ax_xlim(axe_p, xlim_p):
	axe_p.set_xlim(xlim_p)

def AV_ax_xgrid(axe_p, isOn=True):
	axe_p.xaxis.grid(isOn)

def AV_ax_axhline(axe_p, y_p, xmin_p, xmax_p):
# Add a horizontal line across the axis

	axe_p.axhline(y=y_p, xmin=xmin_p, xmax=xmax_p)

########################### Y-axis's properties ###########################
def AV_ax_ylabel(axe_p, ylabel_p, ylabel_size_p=20):
	axe_p.set_ylabel(ylabel_p, fontsize=ylabel_size_p)

def AV_ax_ylabel_fontsizel(axe_p, ylabel_size_p=20):
	axe_p.set_ylabel(axe_p.get_ylabel(), fontsize=ylabel_size_p)


def AV_ax_yscale(axe_p, yscale_p, yscale_base_p=10):
	if (yscale_p == "linear"):
		axe_p.set_yscale(yscale_p)
	elif (yscale_p == "log"):
		axe_p.set_yscale(yscale_p, basey=yscale_base_p)

def AV_ax_ytick(axe_p, ytick_p, minor_p=False):
	if (type(ytick_p).__name__ == "list") & (np.shape(ytick_p)[0] == 0):
		# This is true when ytick_p = []
		axe_p.yaxis.set_major_locator(plt.NullLocator())
		axe_p.yaxis.set_minor_formatter(plt.NullFormatter())
	else:
		axe_p.set_yticks(ytick_p, minor=minor_p)
		axe_p.yaxis.set_minor_formatter(plt.NullFormatter())

def AV_ax_yticklabel(axe_p, ytick_label_p):
	if (type(ytick_label_p).__name__ == "list") & (np.shape(ytick_label_p)[0] == 0):
		# This is true when ytick_label_p = []
		axe_p.yaxis.set_major_formatter(plt.NullFormatter())
	else:
		axe_p.set_yticklabels(ytick_label_p)

def AV_ax_yticklabel_fontsize(axe_p, ytick_label_fontsize_p=15):
	for tick_l in axe_p.yaxis.get_major_ticks():
		tick_l.label.set_fontsize(ytick_label_fontsize_p)

def AV_ax_ylim(axe_p, ylim_p):
	axe_p.set_ylim(ylim_p)

def AV_ax_ygrid(axe_p, isOn=True):
	axe_p.yaxis.grid(isOn)

def AV_ax_axvline(axe_p, x_p, ymin_p, ymax_p):
# Add a vertical line across the axis

	axe_p.axvline(x=x_p, ymin=ymin_p, ymax=ymax_p)


def AV_makeFigNice(isShowFig_p=True, \
	fig_p=None, axe_p=None, \
	title_p=None, title_size_p=50, \
	xlabel_p=None, xlabel_size_p=40, \
	xtick_p=None, xtick_label_p=None, xtick_label_fontsize_p= 30, \
	xscale_p=None, xscale_base_p=10, \
	xlim_p=None, \
	ylabel_p=None, ylabel_size_p=40, \
	ytick_p=None, ytick_label_p=None, ytick_label_fontsize_p= 30, \
	yscale_p=None, yscale_base_p=10, \
	ylim_p=None, \
	abs_path_FN_p=None):
# Params:
# 	axe_p is subclass of Axes (https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes)

	axes_l = None
	if axe_p is not None:
		axes_l = axe_p
	else:
		pass
		# if fig_p is not None:
		# 	axes_l = fig_p.gca()

	############### Title ###############
	if title_p is None:
		if fig_p is None:
			axes_l.set_title("")
		else:
			fig_p.suptitle("")
	else:
		if fig_p is None:
			axes_l.set_title(title_p, fontsize=title_size_p)
		else:
			fig_p.suptitle(title_p, fontsize=title_size_p)

	############### x-axis ###############
	if xlabel_p is not None:
		if fig_p is None:
			axes_l.set_xlabel(xlabel_p, fontsize=xlabel_size_p)
		else:
			fig_p.text(0.5, 0.04, xlabel_p, ha='center', fontsize=xlabel_size_p)

	# axes_l.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

	if (xscale_p == "linear"):
		axes_l.set_xscale(xscale_p)
	if (xscale_p == "log"):
		axes_l.set_xscale(xscale_p, basex=xscale_base_p)

	if axe_p is not None:
		for tick_l in axes_l.xaxis.get_major_ticks():
			tick_l.label.set_fontsize(xtick_label_fontsize_p)

	if (xtick_p is not None):
		if (type(xtick_p).__name__ == "list") & (np.shape(xtick_p)[0] == 0):
			# This is true when ytick_p = []
			axes_l.xaxis.set_major_locator(plt.NullLocator())
		else:
			axes_l.set_xticks(xtick_p, minor=False)

	if (xtick_label_p is not None):
		if (type(xtick_label_p).__name__ == "list") & (np.shape(xtick_label_p)[0] == 0):
			# This is true when xtick_label_p = []
			axes_l.xaxis.set_major_formatter(plt.NullFormatter())
		else:
			axes_l.set_xticklabels(xtick_label_p)

	if xlim_p is not None:
		axes_l.set_xlim(xlim_p)
	############### y-axis ###############
	if ylabel_p is not None:
		if fig_p is None:
			axes_l.set_ylabel(ylabel_p, fontsize=ylabel_size_p)
		else:
			fig_p.text(0.04, 0.5, ylabel_p, va='center', rotation='vertical', fontsize=ylabel_size_p)

	# axes_l.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
	

	if (yscale_p == "linear"):
		axes_l.set_yscale(yscale_p)
	if (yscale_p == "log"):
		axes_l.set_yscale(yscale_p, basey=yscale_base_p)

	if axe_p is not None:
		for tick_l in axes_l.yaxis.get_major_ticks():
			tick_l.label.set_fontsize(ytick_label_fontsize_p)

	if (ytick_p is not None):
		if (type(ytick_p).__name__ == "list") & (np.shape(ytick_p)[0] == 0):
			# This is true when ytick_p = []
			axes_l.yaxis.set_major_locator(plt.NullLocator())
		else:
			axes_l.set_yticks(ytick_p, minor=False)

	if (ytick_label_p is not None):
		if (type(ytick_label_p).__name__ == "list") & (np.shape(ytick_label_p)[0] == 0):
			# This is true when ytick_label_p = []
			axes_l.yaxis.set_major_formatter(plt.NullFormatter())
		else:
			axes_l.set_yticklabels(ytick_label_p)

	if ylim_p is not None:
		axes_l.set_ylim(ylim_p)

	##### save figure #####
	if (abs_path_FN_p is not None):
		# AV_show_plot(fig_p=fig_p)

		# plt.show(block=False)
		# plt.pause(1.0)
		# plt.gca().set_axis_off()
		# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
		# plt.margins(0,0)
		# plt.gca().xaxis.set_major_locator(plt.NullLocator())
		# plt.gca().yaxis.set_major_locator(plt.NullLocator())
		# plt.savefig(abs_path_FN_p, bbox_inches = 'tight', pad_inches = 0)

		##### Maximize #####
		plt.figure(num=fig_p.number)
		plt.pause(1.0)
		graphic_backend_l = plt.get_backend().lower()
		if ((graphic_backend_l == "qt4agg") | (graphic_backend_l == "qt5agg")):
			mng_l = plt.get_current_fig_manager()  # Open directly in full window

			mng_l.window.showMaximized()
		elif (graphic_backend_l == "wxagg"):
			mng_l = plt.get_current_fig_manager()  # Open directly in full window

			mng_l.frame.Maximize(True)
		elif (graphic_backend_l == "tkagg"):
			mng_l = plt.get_current_fig_manager()  # Open directly in full window

			mng_l.resize(*mng_l.window.maxsize())

		plt.show(block=False)
		plt.pause(1.0)
		print(abs_path_FN_p)
		fig_p.savefig(abs_path_FN_p, bbox_inches = 'tight', pad_inches = 0)

		if not isShowFig_p:
			plt.close(fig_p)

########################### Colorbar ###########################
def AV_ax_get_cbar(axe_p, c_p, ticks_p=None, ticks_label_p=None, fontsize_p=20):
	if (ticks_p is None):
		cbar_l = AV_ax_get_fig(axe_p=axe_p).colorbar(c_p)
	else:
		cbar_l = AV_ax_get_fig(axe_p=axe_p).colorbar(c_p, ticks=ticks_p)

		cbar_l.ax.set_yticklabels(ticks_label_p, fontsize=fontsize_p)


	return cbar_l

def AV_ax_cbar_set_ticks(axe_p, c_p, ticks_p):
	AV_ax_get_fig(axe_p=axe_p).colorbar(c_p, ax=axe_p, ticks=ticks_p)

def AV_ax_cbar_set_text(cbar_p, text_p, fontsize_p=20):
	cbar_p.ax.set_ylabel(text_p, fontsize=fontsize_p)

################################ Multi-cores ################################
mtc_lock_g = None

def AV_init(mtc_lock_p):
    global mtc_lock_g
    mtc_lock_g = mtc_lock_p

def AV_multicore_oneGPU_main(func_p, paramsList_p):

	global mtc_lock_g

	mtc_lock_g = multiprocessing.Lock()
	with Pool(processes=AV_get_N_cores(), initializer=AV_init, initargs=(mtc_lock_g,)) as pool:
		print("AV_multicore_main: #cores =" + str(AV_get_N_cores()))
		resultList_l = pool.starmap(func_p, paramsList_p)

	return resultList_l


def AV_multicore_makeParams(paramsList_p, *args):
# Params:
# 	paramsList_p: a list of parameters given to a function.
# 	*args: arguments. Note that we have to list all arguments of func_p in order of their appearance of func_p. 

	param_l = []
	for arg in args:
		param_l.append(arg)

	paramsList_p.append(param_l)

	return paramsList_p

def AV_multicore_main(func_p, paramsList_p, mtp_start_method_p=None, isDebug_p=True):
	"""
	This will wait for all tasks to be completed and return the control to the caller. paramsList_p=[(1,2), (3, 4)] results in [func_p(1,2), func_p(3,4)], i.e. it preserves order.
	Params:
		func_p: the name of the function.
		paramsList_p: the list of parameters created by AV_multicore_makeParams.
	"""

	if (mtp_start_method_p is not None):
		# https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_start_method
		# The parent process starts a fresh python interpreter process. The child process will only inherit those resources necessary to run the process objects run() method. In particular, unnecessary file descriptors and handles from the parent process will not be inherited. Starting a process using this method is rather slow compared to using fork or forkserver.
		# To avoid "pycuda._driver.LogicError: cuInit failed: initialization error", we set mtp_start_method_p="spawn"
		try:
			multiprocessing.set_start_method(mtp_start_method_p)
		except RuntimeError:
			pass

	# with Pool(processes=N_cores_g) as pool:
	# 	print("AV_multicore_main: #cores =" + str(N_cores_g))
	# 	resultList_l = pool.starmap(func_p, paramsList_p)

	if isDebug_p:
		print("AV_multicore_main: #cores =" + str(AV_get_N_cores()))

	pool_l = Pool(AV_get_N_cores())
	resultList_l = pool_l.starmap(func_p, paramsList_p)
	pool_l.close()
	pool_l.join()
		
	return resultList_l

def AV_multicore_mainV2(func_p, paramsList_p, mtp_start_method_p=None):
	"""
	This will wait for all tasks to be completed and return the control to the caller
	Params:
		func_p: the name of the function.
		paramsList_p: the list of parameters created by AV_multicore_makeParams.
	"""

	try:
		N_params_l = len(paramsList_p)

		print("AV_multicore_mainV2: There are " + str(N_params_l) + " parameters.")

		# Serialize the parameters.
		for i_l in range(N_params_l):
			with open("tmp" + str(i_l), "wb+") as fp:
				pickle.dump(paramsList_p[i_l], fp)
				fp.close()

		if (mtp_start_method_p is not None):
			# https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_start_method
			# The parent process starts a fresh python interpreter process. The child process will only inherit those resources necessary to run the process objects run() method. In particular, unnecessary file descriptors and handles from the parent process will not be inherited. Starting a process using this method is rather slow compared to using fork or forkserver.
			# To avoid "pycuda._driver.LogicError: cuInit failed: initialization error", we set mtp_start_method_p="spawn"
			multiprocessing.set_start_method(mtp_start_method_p)

		id_lin_l = np.linspace(0, N_params_l, AV_get_N_cores() + 1)
		for i_l in range(AV_get_N_cores()):	
			p_l = Process(target=func_p, args=(i_l, int(id_lin_l[i_l]), int(id_lin_l[i_l + 1])))
			p_l.start()

		while not all_job_done(N_cores_p=AV_get_N_cores()):
			time.sleep(1)

		del_job_done(N_cores_p=AV_get_N_cores())

	# Delete the parameters.
	finally:
		del_job_done(N_cores_p=AV_get_N_cores())

		for i_l in range(N_params_l):
			AV_delFile("tmp" + str(i_l))

	# Here we need to combine all results.
	return 0

def all_job_done(N_cores_p):

	for i_l in range(N_cores_p):
		if not AV_isFileExist("tmp" + str(i_l) + "done"):
			return False

	return True

def del_job_done(N_cores_p):

	for i_l in range(N_cores_p):
		AV_delFile("tmp" + str(i_l) + "done")

def AV_get_N_cores():   

	with open(AV_norm_path(AV_smb_absPathToWork()[0] + "/UtilSrcCode/python-Helper/N_cores_g_profiles"), "rb") as fp:
		all_clients_N_cores_g = pickle.load(fp)
		fp.close()

	if AV_is_vattha():
		print("vattha")
		N_cores_g = all_clients_N_cores_g["vattha"]

	elif AV_is_vattha_mac():
		print("vattha_mac")
		N_cores_g = all_clients_N_cores_g["vattha_mac"]

	elif AV_is_pre():
		print("pre")
		N_cores_g = all_clients_N_cores_g["pre"]

	elif AV_is_thanaphon():
		print("thanaphon")
		N_cores_g = all_clients_N_cores_g["thanaphon"]

	elif AV_is_yok():
		print("yok")
		N_cores_g = all_clients_N_cores_g["yok"]

	elif AV_is_nam():
		print("nam")
		N_cores_g = all_clients_N_cores_g["nam"]

	return N_cores_g

################################ Discretize spaces ################################
def AV_1Dspace(val_begin_p, val_end_p, N_val_p, typeOfSampling_p=0, logspaceBase_p=10.0):
    if (typeOfSampling_p == 0):
        return np.linspace(val_begin_p, val_end_p, N_val_p)        
    elif (typeOfSampling_p == 1):        
        return np.logspace(math.log(val_begin_p, logspaceBase_p), math.log(val_end_p, logspaceBase_p), num=N_val_p, base=logspaceBase_p)
    else:
        pass

################################ Test functions ################################
def test_AV_plot_multiset_stackedbar():

	N_subjects_l = 3
	N_conds_l = 2
	N_stacks_l = 3

	dims_l = (N_subjects_l, N_conds_l, N_stacks_l)
	data_l = AV_NaNs(dims_p=dims_l)

	for i_l in range(N_subjects_l):
		for j_l in range(N_conds_l):
			for k_l in range(N_stacks_l):
				data_l[i_l, j_l, k_l] = i_l + j_l + k_l


	fig, ax = plt.subplots()

	AV_plot_multiset_stackedbar(axe_p=ax, \
		data_p=data_l, \
		labels_subjects_p=["sub1", "sub2", "sub3"], \
		labels_subjects_size_p=30, \
		colors_conds_p=[AV_colors("blue"), AV_colors("red")], \
		labels_conds_p=["con1", "con2"], \
		labels_conds_size_p=20, \
		patterns_stacks_p=[AV_patterns("backslash"), AV_patterns("slash"), AV_patterns("dot")], \
		isShow_labels_subjects_p=True, \
		labels_stacks_p=["stack1", "stack2", "stack3"])


	# legend_elements_l = []
	# e_l = AV_ax_legend_elements(type_p=2, label_p="stack1", color_p=AV_colors("white"), pattern_p=AV_patterns("backslash"))
	# legend_elements_l.append(e_l)
	# e_l = AV_ax_legend_elements(type_p=2, label_p="stack2", color_p=AV_colors("white"), pattern_p=AV_patterns("slash"))
	# legend_elements_l.append(e_l)
	# e_l = AV_ax_legend_elements(type_p=2, label_p="stack3", color_p=AV_colors("white"), pattern_p=AV_patterns("dot"))
	# legend_elements_l.append(e_l)

	
	# # legend_elements_l = [Line2D([0], [0], color='b', lw=4, label='Line'),
 # #                   Line2D([0], [0], marker='o', color='w', label='Scatter',
 # #                          markerfacecolor='g', markersize=15),
 # #                   Patch(facecolor='orange', edgecolor='r',
 # #                         label='Color Patch')]	
	# print(legend_elements_l)                   
	# AV_ax_legend(axe_p=ax, legend_elements_p=legend_elements_l, size_p=40)

	plt.show()


def test_AV_plot_heatmap():
	fig, ax = plt.subplots()
	Z = np.random.random(size=(7,10))
	x = 10.0 ** np.arange(10)
	y = 2.0 ** np.arange(7)

	# x = np.arange(10)
	# y = np.arange(7)

	for i_x in np.arange(10):
		for i_y in np.arange(7):
			Z[i_y, i_x] = i_x + i_y

	# Z[0,2] = np.nan
	# Z[0,1] = np.nan
	# Z[0,3] = np.nan

	# Z[1,2] = np.nan
	# Z[1,1] = np.nan
	# Z[1,3] = np.nan

	# print(np.max(Z))
	# print(np.min(Z))

	c_r = AV_plot_heatmap(fig, axe_p=ax, x_p=x, y_p=y, C_p=Z, \
		is_sym_cmap_p=False, \
		shading_p="gouraud", \
		# shading_p="flat", \
		xscale_p="log", xscale_base_p=10, \
		# xscale_p="linear", xscale_base_p=10, \
		yscale_p="log", yscale_base_p=2, \
		# yscale_p="linear", yscale_base_p=2, \
		clamped_C_max_p=14, clamped_C_min_p=1)

	cbar_r = AV_ax_get_cbar(axe_p=ax, c_p=c_r, ticks_p=[1, 14], ticks_label_p=["1", "14"])
	AV_ax_cbar_set_text(cbar_p=cbar_r, text_p="Test", fontsize_p=20)

	# AV_ax_get_fig(axe_p=ax).colorbar(c_r, ax=ax, ticks=[8, 14])
	# cbar = AV_ax_get_fig(axe_p=ax).colorbar(c_r)
	# cbar.ax.set_ylabel('verbosity coefficient')

	# AV_ax_xscale(axe_p=ax, xscale_p="log")
	# AV_ax_yscale(axe_p=ax, yscale_p="log")

	# plt.colorbar()

	# colorbar(c, ax=ax)

	AV_fig_maximized(fig_p=fig)

	plt.show()

def test_AV_radar_chart_MUSE():

	AV_radar_chart_MUSE(ax_p=plt.subplot(111, polar=True), chns_label_p=["TP9", "AF7", "AF8", "TP10"], N_circles_p=3)

	plt.show()

if __name__ == "__main__":

    x = np.array([[[0,0,1],[2,0,3]],[[4,0,5],[6,0,7]]])

    x[0,0,0] = 10
    x[0,0,1] = 100
    x[0,0,2] = 200

    x[1,0,0] = 11
    x[1,0,1] = 101
    x[1,0,2] = 201

    print(np.shape(x))
    print(AV_swapaxes(x, 0, 2))
    y = AV_swapaxes(x, 1, 2)

    print(y[1, 1, 0])

	# test_AV_radar_chart_MUSE()

	# test_AV_plot_multiset_stackedbar()

	# AV_plot_allcmap()

	# s0 = np.zeros((1,1))
	# s1 = np.zeros((1,2))
	# s2 = np.zeros((2,1))
	# s3 = np.zeros((2,2))	

	# paramsList_l = []
	# paramsList_l = AV_multicore_makeParams(paramsList_l, True, s1, s2, s3)
	# paramsList_l = AV_multicore_makeParams(paramsList_l, s0)

	# # print(paramsList_l)

 #    # # print(AV_1Dspace(1, 100, 100, typeOfSampling_p=1))
 #    # aList = [123, 'xyz', 'zara', 'abc']
 #    # aTuple = tuple(aList)
 #    # # print(aTuple)
 #    # freeze_support()
	# AV_multicore_main(func, paramsList_l)    
 #    # # test_var_kwargs(farg=1, myarg2="two", myarg3=3)
 #    # # test_var_kwargs(1,2,3)
 #    # print("fdfdf")
 #    # func(np.zeros((1,2)), np.zeros((1,1)))

 #    # list_l = []
 #    # print(list_l)
 #    # list_l.append(np.zeros((1,2)))
 #    # print(list_l)
 #    # list_l.append(np.zeros((1,1)))
 #    # print(tuple(list_l))
 #    # AV_multicore_makeParams(np.zeros((1,2)), np.zeros((1,1)))
