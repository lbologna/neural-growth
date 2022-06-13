"""
Script for reading all .mat files containing the spike trains and storing them
into a json dictionary. Source .mat files contain the entire recording (
previoulsy obtained by contatenation of several chunks)
"""

import neo 
import pprint
from scipy.io import loadmat
import scipy
import plotly
import plotly.graph_objects as go
import os
import json
from collections import OrderedDict


# set folder/file names
input_folder = '/Users/lbologna/MyWork/2005-2009_phd/data_backup_temp/' + \
    'Data/all_exp/ExpXpapModelFull/'
output_folder = './'
output_file = "data_all_except_one_div.json"

all_data = OrderedDict()
sf = 10000.0

# explore directory tree to fetch ptrain data
for root, d_names, f_names in os.walk(input_folder):
    root_split = root.split(os.path.sep)
    if "PeakDetectionMAT" in root_split[-2] and "ptrain_" in root_split[-1] \
        and root_split[-1][0:7] == "ptrain_":
        for batch_i in root_split:
            if 'batch' in batch_i and batch_i not in all_data.keys():
                all_data[batch_i] = OrderedDict()
                batch = batch_i
                break
        folder = root_split[-1]
        dir_split = folder.split("_")
        exp_name = dir_split[1]
        if str(exp_name) not in all_data[batch].keys():
            all_data[batch][exp_name] = OrderedDict()
        for j in dir_split:
            if "div" in j.lower() and \
                    j not in all_data[batch][exp_name].keys():
                div = j.lower()
                all_data[batch][exp_name][div] = OrderedDict() 
                all_data[batch][exp_name][div]["chs"] = OrderedDict() 
                break

        #
        print("Collecting data for batch - exp - div ", batch, exp_name, div)
        rec_len = -1
        for f in f_names:
            timestamps = []
            try:
                annots = loadmat(os.path.join(root, f))
                peak_train = annots['peak_train']
                crr_rec_len = peak_train.shape[0]
                if rec_len == -1:
                    rec_len = crr_rec_len
                else:
                    if rec_len != crr_rec_len:
                        raise Exception ('difference in recording times')
                ch = f[-6:-4]
                idx = scipy.sparse.find(annots['peak_train'])[0]
                for ts in idx:
                    timestamps.append(ts/sf)
                    
            except:
                print("ERROR: unable to read file: " + f)
            if len(timestamps) != 0:
                all_data[batch][exp_name][div]["chs"][ch] = timestamps
        if not bool(all_data[batch][exp_name][div]):
            all_data[batch][exp_name].pop(div, None)
        else:
            all_data[batch][exp_name][div]["rec_len"] = rec_len/sf


with open(os.path.join(output_folder, output_file), 'w') as fp:
    json.dump(all_data, fp)
