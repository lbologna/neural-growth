import sys
import os
import json
import plotly.graph_objects as go
import plotly
import numpy as np
import pprint
from collections import OrderedDict
import plotly.express as px

with open(sys.argv[1], 'r') as fp:
    stats = json.load(fp)



def compute_os(stats):
    mfr_all = []
    div_all = []
    os_stats = {
        "batches": {}
    }
    for b in stats["batches"].keys():
        print("##############")
        print("##############")
        print(b)
        if b not in os_stats["batches"].keys():
            os_stats["batches"][b] = {}
            for e in stats["batches"][b].keys():
                if e not in os_stats["batches"][b].keys():
                    os_stats["batches"][b][e] = {}
                    divs = []
                    mfr_mean = []
                    mfr_sem = []
                    mbr_mean = []
                    mbr_sem = []
                    mfib_mean = []
                    mfib_sem = []
                    mburdur_mean = []
                    mburdur_sem = []
                    mnrs_mean = []
                    mnrs_sem = []
                    divs = stats["batches"][b][e].keys()
                    divs_num = [int(i.lower().replace("div","")) for i in divs]
                    crr_mfr = []
                    crr_div = []
                    for d in divs:
                        crr_div_int = int(d.lower().replace("div",""))
                        crr_mfr.append(stats["batches"][b][e][d]["mfr"]["mean"])
                        crr_div.append(crr_div_int)
                        #
                        mfr_mean.append(stats["batches"][b][e][d]["mfr"]["mean"])
                        mfr_sem.append(stats["batches"][b][e][d]["mfr"]["sem"])
                        #
                        mbr_mean.append(stats["batches"][b][e][d]["mbr"]["mean"])
                        mbr_sem.append(stats["batches"][b][e][d]["mbr"]["sem"])
                        #
                        mfib_mean.append(stats["batches"][b][e][d]["mfib"]["mean"])
                        mfib_sem.append(stats["batches"][b][e][d]["mfib"]["sem"])
                        #
                        mburdur_mean.append(stats["batches"][b][e][d]["mburdur"]["mean"])
                        mburdur_sem.append(stats["batches"][b][e][d]["mburdur"]["sem"])
                        #
                        mnrs_mean.append(stats["batches"][b][e][d]["mnrs"]["mean"])
                        mnrs_sem.append(stats["batches"][b][e][d]["mnrs"]["sem"])

                        #
                    #div_int = int(d.lower().replace("div",""))
                    #mfr_all.append(stats["batches"][b][e][d]["mfr"]["mean"])

                    min_val = min(crr_mfr)
                    max_val = max(crr_mfr)
                    print(min_val)
                    print(max_val)
                    crr_mfr_n = [(x-min_val)/(max_val-min_val) for x in crr_mfr]
                    print(crr_mfr_n)
                    for i in range(len(crr_mfr)):
                        mfr_all.append(crr_mfr_n[i])
                        div_all.append(crr_div[i])

                    sort_index = np.argsort(np.array(divs_num))
                    divs = [divs[i] for i in sort_index]
                    #
                    mfr_mean = [mfr_mean[i] for i in sort_index]
                    mfr_sem = [mfr_sem[i] for i in sort_index]
                    #
                    mbr_mean = [mbr_mean[i] for i in sort_index]
                    mbr_sem = [mbr_sem[i] for i in sort_index]
                    #
                    mfib_mean = [mfib_mean[i] for i in sort_index]
                    mfib_sem = [mfib_sem[i] for i in sort_index]
                    #
                    mburdur_mean = [mburdur_mean[i] for i in sort_index]
                    mburdur_sem = [mburdur_sem[i] for i in sort_index]
                    #
                    mnrs_mean = [mnrs_mean[i] for i in sort_index]
                    mnrs_sem = [mnrs_sem[i] for i in sort_index]
                    #
                    idx_os = np.argmax(mfr_mean)
                    mfr_ss_mean = np.nanmean(np.array(mfr_mean[-3:]), dtype=np.float64)
                    #
                    idx_os_mbr = np.argmax(mbr_mean)
                    mbr_ss_mean = np.nanmean(np.array(mbr_mean[-3:]), dtype=np.float64)
                    #
                    idx_os_mfib = np.argmax(mfib_mean)
                    mfib_ss_mean = np.nanmean(np.array(mfib_mean[-3:]), dtype=np.float64)
                    #
                    idx_os_mburdur = np.argmax(mburdur_mean)
                    mburdur_ss_mean = np.nanmean(np.array(mburdur_mean[-3:]), dtype=np.float64)
                    #
                    idx_os_mnrs = np.argmax(mnrs_mean)
                    mnrs_ss_mean = np.nanmean(np.array(mnrs_mean[-3:]), dtype=np.float64)
                    print("--------")
                    print(e)
                    print(divs)
                    perc_mfr = 100*(mfr_mean[idx_os]-mfr_ss_mean)/mfr_ss_mean
                    perc_mbr = 100*(mbr_mean[idx_os_mbr]-mbr_ss_mean)/mbr_ss_mean
                    perc_mfib = 100*(mfib_mean[idx_os_mfib]-mfib_ss_mean)/mfib_ss_mean
                    perc_mburdur = 100*(mburdur_mean[idx_os_mburdur]-mburdur_ss_mean)/mburdur_ss_mean
                    perc_mnrs = 100*(mnrs_mean[idx_os_mnrs]-mnrs_ss_mean)/mnrs_ss_mean
                    os_stats["batches"][b][e]["perc_mfr"] = perc_mfr
                    os_stats["batches"][b][e]["perc_mbr"] = perc_mbr
                    os_stats["batches"][b][e]["perc_mfib"] = perc_mfib
                    os_stats["batches"][b][e]["perc_mburdur"] = perc_mburdur
                    os_stats["batches"][b][e]["perc_mnrs"] = perc_mnrs
        
    #pprint.pprint(os_stats)
    perc_mfr_all_x = []
    perc_mfr_all_y = []
    for b in os_stats["batches"]:
        for eidx, e in enumerate(os_stats["batches"][b].keys()):
            perc_mfr_all_y.append(os_stats["batches"][b][e]["perc_mfr"])
            perc_mfr_all_x.append(b + "-" + str(eidx))
    fig = px.bar(x=perc_mfr_all_x, y=perc_mfr_all_y, labels={'x':'Batch names', 'y':'%DeltaMFR'})
    fig.show()
    fig.write_image("overshoot_mean.pdf")


    sort_index = np.argsort(div_all)
    div_ord = [div_all[i] for i in sort_index]
    mfr_ord = [mfr_all[i] for i in sort_index]
    # calculate polynomial
    z = np.polyfit(div_ord, mfr_ord, 3)
    f = np.poly1d(z)
    print f

    # calculate new x's and y's
    x_new = np.linspace(div_ord[0], div_ord[-1], 50)
    y_new = f(x_new)


    fig = go.Figure()
    fig.add_traces(go.Scatter(x = div_all, y = mfr_all, type="scatter", mode="markers"))
    fig.add_traces(go.Scatter(x = x_new, y = y_new, type="scatter", mode="lines"))
    fig.show()



os_stats = compute_os(stats)
