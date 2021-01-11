import os
import json
import plotly.graph_objects as go
import plotly
import numpy as np
import pprint
from collections import OrderedDict

spxb = 5 # minimum number of spikes per burst
isi = 0.1 # maximum distance between spikes in burst (ms) 
min_ch = 0 # minimum number of firing channels
min_sp = 5 # minimum number of spikes
min_mbr = 1 # minimum mean bursting rate

data_file = "data.json"
output_file = "stats.json"

def compute_stats(data, stats):

    # initialize
    mfr = np.array([])
    mbr = np.array([])
    mfib = np.array([])
    mburdur = np.array([])
    mnrs = np.array([])
   
    rec_len = stats["rec_len"]

    for ch in data["chs"].keys():
        ch_bursts = []
        ch_nrs = []
        ch_mfib = np.array([])
        ch_mburdur = np.array([])
        ch_sp_in_bursts = 0
        ts =  data["chs"][ch]
        num_sp = len(ts)
        if num_sp > min_sp:
            ch_mfr = num_sp/stats["rec_len"]
            stats["mfr"]["chs"][ch] = ch_mfr
            mfr = np.append(mfr, ch_mfr)

            # compute difference between adjacent timestamp and their indices
            diff_list = np.diff(ts)
            diff_idx = np.nonzero(diff_list > isi)[0]

            # populate channel burst array
            if diff_idx[0] >= spxb - 1:
                crr_burst = ts[0:diff_idx[0]+1]
                crr_burst_dur =  crr_burst[-1] - crr_burst[0]
                ch_bursts.append(crr_burst)
                ch_mfib = np.append(ch_mfib, len(crr_burst)/crr_burst_dur)
                ch_mburdur = np.append(ch_mburdur, crr_burst_dur)
                ch_sp_in_bursts += len(crr_burst)
            for pos in range(1, len(diff_idx)):
                if diff_idx[pos] - diff_idx[pos-1] - 1 >= spxb:
                    crr_burst = ts[diff_idx[pos-1]+1:diff_idx[pos]+1]
                    crr_burst_dur =  crr_burst[-1] - crr_burst[0]
                    ch_bursts.append(crr_burst)
                    ch_mfib = np.append(ch_mfib, len(crr_burst)/crr_burst_dur)
                    ch_mburdur = np.append(ch_mburdur, crr_burst_dur)
                    ch_sp_in_bursts += len(crr_burst)
            if (len(ts) - diff_idx[-1] - 1) >= spxb:
                crr_burst = ts[diff_idx[-1]+1:]
                crr_burst_dur =  crr_burst[-1] - crr_burst[0]
                ch_bursts.append(crr_burst)
                ch_mfib = np.append(ch_mfib, len(crr_burst)/crr_burst_dur)
                ch_mburdur = np.append(ch_mburdur, crr_burst_dur)
                ch_sp_in_bursts += len(crr_burst)

            #
            if len(ch_bursts)>0:
                ch_mbr = 60 * float(len(ch_bursts))/rec_len
                if ch_mbr > min_mbr:

                    ch_mfib_mean = np.nanmean(ch_mfib, dtype=np.float64)
                    ch_mfib_sem = np.nanstd(ch_mfib, dtype=np.float64) / \
                        np.sqrt(np.size(ch_mfib))

                    ch_mburdur_mean = np.nanmean(ch_mburdur, dtype=np.float64)
                    ch_mburdur_sem = np.nanstd(ch_mburdur, dtype=np.float64) / \
                        np.sqrt(np.size(ch_mburdur))

                    ch_mnrs = (len(ts) - ch_sp_in_bursts) / float(len(ts))
                else:
                    ch_mfib_mean = 0
                    ch_mfib_sem = 0

                    ch_mburdur_mean = 0
                    ch_mburdur_sem = 0

                    ch_mnrs = 1

                # 
                stats["mbr"]["chs"][ch] = ch_mbr
                #
                stats["mfib"]["chs"][ch] = {}
                stats["mfib"]["chs"][ch]["mean"] = ch_mfib_mean 
                stats["mfib"]["chs"][ch]["sem"] = ch_mfib_sem
                #
                stats["mburdur"]["chs"][ch] = {}
                stats["mburdur"]["chs"][ch]["mean"] = ch_mburdur_mean
                stats["mburdur"]["chs"][ch]["sem"] = ch_mburdur_sem 
                #
                stats["mnrs"]["chs"][ch] = ch_mnrs

                mbr = np.append(mbr, ch_mbr)
                mfib = np.append(mfib, ch_mfib_mean)
                mburdur = np.append(mburdur, ch_mburdur_mean)
                mnrs = np.append(mnrs, ch_mnrs)

    if np.size(mfr) > min_ch:
        stats["mfr"]["mean"] = np.nanmean(mfr, dtype=np.float64)
        stats["mfr"]["sem"] = np.nanstd(mfr, dtype=np.float64) / \
                np.sqrt(np.size(mfr))

        if np.size(mbr) > 0:
            stats["mbr"]["mean"] = np.nanmean(mbr, dtype=np.float64)
            stats["mbr"]["sem"] = np.nanstd(
                mbr, dtype=np.float64) / np.sqrt(np.size(mbr)
            )

            stats["mfib"]["mean"] = np.nanmean(mfib, dtype=np.float64)
            stats["mfib"]["sem"] = np.nanstd(
                mfib, dtype=np.float64) / np.sqrt(np.size(mfib)
            )

            stats["mburdur"]["mean"] = np.nanmean(mburdur, dtype=np.float64)
            stats["mburdur"]["sem"] = np.nanstd(
                mburdur, dtype=np.float64) / np.sqrt(np.size(mburdur)
            )

            stats["mnrs"]["mean"] = np.nanmean(mnrs, dtype=np.float64)
            stats["mnrs"]["sem"] = np.nanstd(
                mnrs, dtype=np.float64) / np.sqrt(np.size(mnrs)
            )
        else:
            stats["mbr"]["mean"] = 0
            stats["mbr"]["sem"] = 0

            stats["mfib"]["mean"] = 0
            stats["mfib"]["sem"] = 0

            stats["mburdur"]["mean"] = 0
            stats["mburdur"]["sem"] = 0

            stats["mnrs"]["mean"] = 0
            stats["mnrs"]["sem"] = 0

    return
    
#
with open(data_file, 'r') as fp:
    data = json.load(fp)

all_stats = {}
for batch in data.keys():
    print("###########")
    print("Batch " + batch)
    all_stats[batch] = {}
    for exp in data[batch].keys():
        all_stats[batch][exp] = {}
        print("Exp " + exp)
        for div in data[batch][exp].keys():
            ast = all_stats[batch][exp][div] = {}
            all_stats[batch][exp][div]["rec_len"] = \
                data[batch][exp][div]["rec_len"]
            #
            all_stats[batch][exp][div]["mfr"] = {}
            all_stats[batch][exp][div]["mfr"]["mean"] = 0
            all_stats[batch][exp][div]["mfr"]["sem"] = 0
            all_stats[batch][exp][div]["mfr"]["chs"] = {}
            #
            all_stats[batch][exp][div]["mbr"] = {}
            all_stats[batch][exp][div]["mbr"]["mean"] = 0
            all_stats[batch][exp][div]["mbr"]["sem"] = 0
            all_stats[batch][exp][div]["mbr"]["chs"] = {}
            #
            all_stats[batch][exp][div]["mfib"] = {}
            all_stats[batch][exp][div]["mfib"]["mean"] = 0
            all_stats[batch][exp][div]["mfib"]["sem"] = 0
            all_stats[batch][exp][div]["mfib"]["chs"] = {}
            #
            all_stats[batch][exp][div]["mnrs"] = {}
            all_stats[batch][exp][div]["mnrs"]["mean"] = 0
            all_stats[batch][exp][div]["mnrs"]["sem"] = 0
            all_stats[batch][exp][div]["mnrs"]["chs"] = {}
            #
            all_stats[batch][exp][div]["mburdur"] = {}
            all_stats[batch][exp][div]["mburdur"]["mean"] = 0
            all_stats[batch][exp][div]["mburdur"]["sem"] = 0
            all_stats[batch][exp][div]["mburdur"]["chs"] = {}
            compute_stats(data[batch][exp][div], ast)

#with open('results_new.json', 'w') as fp:
#    json.dump(all_stats, fp, sort_keys=True, indent=4)

with open(output_file, 'w') as fp:
    json.dump(all_stats, fp)

