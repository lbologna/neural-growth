import neo 
import pprint
from scipy.io import loadmat
import scipy
import plotly
import plotly.graph_objects as go
import os
import json
import glob
from collections import OrderedDict

#input_folder = '/Volumes/DataLab/Data/all_exp/ExpXpapModelFull'
input_folder = \
    '/Users/lbologna/MyWork/2005-2009_phd/data_backup_temp/' + \
    'Data/all_exp/ExpXpapModel/'
output_folder = './'
output_file = "data_reconstruct_one_div.json"

with open("missing_chunks.json", "r") as f:
    chunks = json.load(f)


all_data = OrderedDict()
sf = 10000.0

b_keys = chunks.keys()
for b in b_keys:
    e_keys = chunks[b].keys()
    for e in e_keys:
        d_keys = chunks[b][e].keys()
        for d in d_keys:
            print("Collecting data for batch - exp - div ", b, " ", e, " ", d)
            # create dicts
            all_data[b] = {}
            all_data[b][e] = {}
            all_data[b][e][d] = {}
            all_data[b][e][d]["chs"] = OrderedDict()

            # 
            pf_path = os.path.join(input_folder, b, e, d) + "/*PeakDetection*"
            peak_folder = glob.glob(pf_path)
            ptrain_all = glob.glob(peak_folder[0] + "/ptrain_*")

            # 
            rec_len = -1

            # extract data chunks from .mat files
            chunk_edges = chunks[b][e][d]
            for idx_ce, ce in enumerate(chunk_edges):
                for pt_fold in ptrain_all:
                    idx_sep = pt_fold.rfind("_")
                    idx_int = int(pt_fold[idx_sep+1:])
                    if ce == idx_int:
                        mat_files = glob.glob(pt_fold + "/ptrain*.mat")
                        for mf in mat_files:
                            timestamps = []
                            try: 
                                annots = loadmat(mf)
                                peak_train = annots['peak_train']
                                crr_rec_len = peak_train.shape[0]
                                if rec_len == -1:
                                    rec_len = crr_rec_len
                                else:
                                    if rec_len != crr_rec_len:
                                        raise Exception (
                                            'Difference in recording times')
                                ch = mf[-6:-4]
                                idx = scipy.sparse.find(
                                    annots['peak_train'])[0]
                                for ts in idx:
                                    timestamps.append(
                                        idx_ce*rec_len/sf + ts/sf)
                                if len(timestamps) != 0:
                                    if ch not in all_data[b][e][d]["chs"]:
                                        all_data[b][e][d]["chs"][ch] = \
                                            timestamps
                                    else:
                                        for tt in timestamps:
                                            all_data[b][e][d]["chs"][
                                                ch].append(tt)
                            except:
                                print("ERROR: unable to read file: " + mf)

        if not bool(all_data[b][e][d]["chs"]):
            all_data[b][e].pop(d, None)
        else:
            all_data[b][e][d]["rec_len"] = rec_len * len(chunk_edges) / sf


with open(os.path.join(output_folder, output_file), 'w') as fp:
    json.dump(all_data, fp)
