import sys
import os
import json
import plotly.graph_objects as go
import numpy as np
import time

def plot_fig(mean, sem, divs, yaxis_title, b, e, stat_label, ch_num=[]):
    filename = b + "_exp_" + e + "_" + stat_label
    title = b + " - exp: " + e + " - " + stat_label + "<br>"
    title += "Number of bursting channels for each div: " 
    for s in ch_num:
        title += " " + str(s)
    fig = go.Figure()
    fig.add_traces(go.Scatter(
        x = divs + divs[::-1],
        y = sem, 
        fill = "tozerox",
        fillcolor = "rgba(0,100,80,0.2)",
        line = dict(color="rgba(0,100,80,0.0)"),
        showlegend = False,
        type = "scatter"))
    fig.add_traces(go.Scatter(
        x = divs,
        y = mean,
        mode= "lines+markers",
        line = dict(color="rgba(0,100,80,1)"),
        showlegend = False,
        type= "scatter"))
    fig.update_layout(title = title, 
        xaxis=dict(range=[divs[0], divs[-1]],title="DIVs"), 
        yaxis=dict(title=yaxis_title))

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    fig.write_html(os.path.join(image_folder, filename + ".html"))


with open(sys.argv[1]) as pf:
    config = json.load(pf)

with open(config["plot_stats"]["stats_file"], 'r') as fp:
    res = json.load(fp)

image_folder = config["plot_stats"]["image_folder"]
wait_time = config["plot_stats"]["wait_time"]



for b in res["batches"].keys():
    for e in res["batches"][b].keys():
        divs = []
        mfr_ch_num = []
        mfr = []
        mfr_sem = []
        mfr_upper = []
        mfr_lower = []

        mbr_ch_num = []
        mbr = []
        mbr_sem = []
        mbr_upper = []
        mbr_lower = []
    
        mfib_ch_num = []
        mfib = []
        mfib_sem = []
        mfib_upper = []
        mfib_lower = []

        mburdur_ch_num = []
        mburdur = []
        mburdur_sem = []
        mburdur_upper = []
        mburdur_lower = []

        mnrs_ch_num = []
        mnrs = []
        mnrs_sem = []
        mnrs_upper = []
        mnrs_lower = []

        for d in res["batches"][b][e].keys():
            div = int(d.lower().replace("div",""))
            divs.append(div)
            mfr.append(res["batches"][b][e][d]["mfr"]["mean"])
            mfr_sem.append(res["batches"][b][e][d]["mfr"]["sem"])
            mfr_ch_num.append(len(res["batches"][b][e][d]["mfr"]["chs"].keys()))

            mbr.append(res["batches"][b][e][d]["mbr"]["mean"])
            mbr_sem.append(res["batches"][b][e][d]["mbr"]["sem"])
            mbr_ch_num.append(len(res["batches"][b][e][d]["mbr"]["chs"].keys()))
            
            mfib.append(res["batches"][b][e][d]["mfib"]["mean"])
            mfib_sem.append(res["batches"][b][e][d]["mfib"]["sem"])
            mfib_ch_num.append(len(res["batches"][b][e][d]["mfib"]["chs"].keys()))
            
            mburdur.append(res["batches"][b][e][d]["mburdur"]["mean"])
            mburdur_sem.append(res["batches"][b][e][d]["mburdur"]["sem"])
            mburdur_ch_num.append(len(res["batches"][b][e][d]["mburdur"]["chs"].keys()))
            
            mnrs.append(res["batches"][b][e][d]["mnrs"]["mean"])
            mnrs_sem.append(res["batches"][b][e][d]["mnrs"]["sem"])
            mnrs_ch_num.append(len(res["batches"][b][e][d]["mnrs"]["chs"].keys()))
        #
        sort_index = np.argsort(np.array(divs))
        
        divs = [divs[i] for i in sort_index]
        mfr_ch_num = [mfr_ch_num[i] for i in sort_index]
        mbr_ch_num = [mbr_ch_num[i] for i in sort_index]
        mfib_ch_num = [mfib_ch_num[i] for i in sort_index]
        mburdur_ch_num = [mburdur_ch_num[i] for i in sort_index]
        mnrs_ch_num = [mnrs_ch_num[i] for i in sort_index]

        #
        mfr = [mfr[i] for i in sort_index]
        mfr_sem = [mfr_sem[i] for i in sort_index]
        mfr_upper = [mfr[i] + mfr_sem[i] for i in range(len(mfr))]
        mfr_lower = [mfr[i] - mfr_sem[i] for i in range(len(mfr))]
        y_mfr_sem = mfr_upper + mfr_lower[::-1]
        
        #
        mbr = [mbr[i] for i in sort_index]
        mbr_sem = [mbr_sem[i] for i in sort_index]
        mbr_upper = [mbr[i] + mbr_sem[i] for i in range(len(mbr))]
        mbr_lower = [mbr[i] - mbr_sem[i] for i in range(len(mbr))]
        y_mbr_sem = mbr_upper + mbr_lower[::-1]
        #
        mfib = [mfib[i] for i in sort_index]
        mfib_sem = [mfib_sem[i] for i in sort_index]
        mfib_upper = [mfib[i] + mfib_sem[i] for i in range(len(mfib))]
        mfib_lower = [mfib[i] - mfib_sem[i] for i in range(len(mfib))]
        y_mfib_sem = mfib_upper + mfib_lower[::-1]
        #
        mburdur = [mburdur[i] for i in sort_index]
        mburdur_sem = [mburdur_sem[i] for i in sort_index]
        mburdur_upper = [mburdur[i] + mburdur_sem[i] 
                for i in range(len(mburdur))]
        mburdur_lower = [mburdur[i] - mburdur_sem[i] 
                for i in range(len(mburdur))]
        y_mburdur_sem = mburdur_upper + mburdur_lower[::-1]
        #
        mnrs = [mnrs[i] for i in sort_index]
        mnrs_sem = [mnrs_sem[i] for i in sort_index]
        mnrs_upper = [mnrs[i] + mnrs_sem[i] for i in range(len(mnrs))]
        mnrs_lower = [mnrs[i] - mnrs_sem[i] for i in range(len(mnrs))]
        y_mnrs_sem = mnrs_upper + mnrs_lower[::-1]
        #
        plot_fig(mfr, y_mfr_sem, divs, "# spikes/s", b, e, "MFR", mfr_ch_num)
        time.sleep(wait_time)
        #
        plot_fig(mbr, y_mbr_sem, divs, "# bursts/min", b, e, "MBR", mbr_ch_num)
        time.sleep(wait_time)
        #
        plot_fig(mfib, y_mfib_sem, divs, "# spikes/burst", b, e, "MFIB",
                 mfib_ch_num)
        time.sleep(wait_time)
        #
        plot_fig(mburdur, y_mburdur_sem, divs, "s", b, e, "MBURDUR", 
                 mburdur_ch_num)
        time.sleep(wait_time)
        #
        plot_fig(mnrs, y_mnrs_sem, divs, "% randoms spikes", b, e, "MNRS", 
                 mnrs_ch_num)
        time.sleep(wait_time)
