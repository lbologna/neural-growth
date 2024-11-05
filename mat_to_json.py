"""
Script for reading all .mat files containing the spike trains and storing them
into a json dictionary. Source .mat files contain the entire recording (
previoulsy obtained by contatenation of several chunks)
"""

import sys
from scipy.io import loadmat
import scipy
import os
import json
from collections import OrderedDict

# set folder/file names
# input_folder = '/Users/lbologna/MyWork/2005-2009_phd/data/' + \
#    '/ExpXpapModelFull/'
# output_folder = './'
# output_file = "data_all_except_one_div.json"
# config_file = "used_recordings_spont.json"

config_file = sys.argv[1]

with open(config_file, 'r') as fp:
    used_rec = json.load(fp)

rec_keys = used_rec.keys()
rec_type = used_rec["fs_config"]["rec_type"]

input_folder = used_rec["fs_config"]["input_folder"] 
output_folder = used_rec["fs_config"]["output_folder"] 
output_file = used_rec["fs_config"]["output_file"] 


all_data_pre = OrderedDict()

if rec_type == "stim_pre_post":    
    all_data_post = OrderedDict()

sf = 10000.0

# explore directory tree to fetch ptrain data
for root, d_names, f_names in os.walk(input_folder):
    root_split = root.split(os.path.sep)
    if "PeakDetectionMAT" in root_split[-2] and "ptrain_" in root_split[-1] \
        and root_split[-1][0:7] == "ptrain_" and root_split[-5] in rec_keys \
        and root_split[-4] in used_rec[root_split[-5]].keys() \
        and root_split[-3].lower() in list(map(str.lower, used_rec[root_split[-5]][root_split[-4]].keys())):
        
        if rec_type == "stim_stim" and "_stim1_" not in root_split[-1]:
            continue
        print("Collecting data for : ", root)
        all_data = OrderedDict()

        # extract info
        batch = root_split[-5]
        folder = root_split[-1]
        dir_split = folder.split("_")
        exp_name = dir_split[1]

        if rec_type == "stim_pre_post":
            if "01_nb_" in folder:
                all_data = all_data_pre
            elif "03_nb_" in folder:
                all_data = all_data_post
        else:
            all_data = all_data_pre

        if batch not in all_data.keys():
            all_data[batch] = OrderedDict()
        
        if str(exp_name) not in all_data[batch].keys():
            all_data[batch][exp_name] = OrderedDict()

        for j in dir_split:
            if "div" in j.lower() and \
                    j not in all_data[batch][exp_name].keys():
                div = j.lower()
                all_data[batch][exp_name][div] = OrderedDict() 
                break
            
        if "_stim1_" in root_split[-1]:
            stim_str_idx = root_split[-1].find("_stim1_")
            stim_ch = root_split[-1][stim_str_idx + 7:stim_str_idx + 9]
            if "stim_ch" not in all_data[batch][exp_name][div].keys():
                all_data[batch][exp_name][div]["stim_ch"] = OrderedDict()
            if stim_ch not in all_data[batch][exp_name][div]["stim_ch"].keys():
                all_data[batch][exp_name][div]["stim_ch"][stim_ch] = OrderedDict()
                all_data[batch][exp_name][div]["stim_ch"][stim_ch]["chs"] = OrderedDict()
        else:
            all_data[batch][exp_name][div]["chs"] = OrderedDict()

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
                if "_stim1_" in root_split[-1]:
                    all_data[batch][exp_name][div]["stim_ch"][stim_ch]["chs"][ch] = timestamps
                else:
                    all_data[batch][exp_name][div]["chs"][ch] = timestamps
        if not bool(all_data[batch][exp_name][div]):
            all_data[batch][exp_name].pop(div, None)
        else:
            all_data[batch][exp_name][div]["rec_len"] = rec_len/sf

        if rec_type == "stim":
            if "01_nb_" in folder:
                all_data_pre = all_data
            elif "03_nb_" in folder:
                all_data_post = all_data
        else:
            all_data_pre = all_data


if rec_type == "stim":
    with open(os.path.join(output_folder, "pre_" + output_file), 'w') as fp:
        json.dump(all_data_pre, fp)
    with open(os.path.join(output_folder, "post_" + output_file), 'w') as fp:
        json.dump(all_data_post, fp)
else:
    with open(os.path.join(output_folder, output_file), 'w') as fp:
        json.dump(all_data_pre, fp)