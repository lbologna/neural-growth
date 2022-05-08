import sys
import os
import json
import plotly.graph_objects as go
import plotly
import numpy as np
import pprint
from collections import OrderedDict
import time
import argparse



def compute_stats(data, stats, config):

    spxb = config["compute_stats"]["spxb"]
    isi = config["compute_stats"]["isi"]
    min_fir_ch = config["compute_stats"]["min_fir_ch"]
    min_brs_ch = config["compute_stats"]["min_brs_ch"]
    min_mfr_ch = config["compute_stats"]["min_mfr_ch"]
    min_mbr_ch = config["compute_stats"]["min_mbr_ch"]

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
        ch_mfr = num_sp/stats["rec_len"]
        print(ch_mfr)

        if ch_mfr > min_mfr_ch:
            stats["mfr"]["chs"][ch] = ch_mfr
            mfr = np.append(mfr, ch_mfr)

            # compute difference between adjacent timestamp and their indices
            diff_list = np.diff(ts)
            diff_idx = np.nonzero(diff_list > isi)[0]
            if len(diff_idx) == 0:
                continue

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
            if len(ch_bursts) > 0:
                ch_mbr = 60 * float(len(ch_bursts))/rec_len
                if ch_mbr > min_mbr_ch:

                    #
                    ch_mfib_mean = np.nanmean(ch_mfib, dtype=np.float64)
                    ch_mfib_sem = np.nanstd(ch_mfib, dtype=np.float64) / \
                        np.sqrt(np.size(ch_mfib))

                    ch_mburdur_mean = np.nanmean(ch_mburdur, dtype=np.float64)
                    ch_mburdur_sem = np.nanstd(ch_mburdur, dtype=np.float64) / \
                        np.sqrt(np.size(ch_mburdur))

                    ch_mnrs = (len(ts) - ch_sp_in_bursts) / float(len(ts))

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

    if np.size(mfr) > min_fir_ch:
        stats["mfr"]["mean"] = np.nanmean(mfr, dtype=np.float64)
        stats["mfr"]["sem"] = np.nanstd(mfr, dtype=np.float64) / \
                np.sqrt(np.size(mfr))

        if np.size(mbr) > min_brs_ch:
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

            stats["mnrs"]["mean"] = 1
            stats["mnrs"]["sem"] = 0
    else:
            stats["mfr"]["mean"] = 0
            stats["mfr"]["sem"] = 0

            stats["mbr"]["mean"] = 0
            stats["mbr"]["sem"] = 0

            stats["mfib"]["mean"] = 0
            stats["mfib"]["sem"] = 0

            stats["mburdur"]["mean"] = 0
            stats["mburdur"]["sem"] = 0

            stats["mnrs"]["mean"] = 1
            stats["mnrs"]["sem"] = 0

    return


def plot_fig(mean, sem, divs, yaxis_title, b, e, stat_label, ch_num=[], image_folder=""):
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
    print(image_folder)
    fig.write_html(os.path.join(image_folder, filename + ".html"))


def plot_stats(stats_file, config, image_folder):

    with open(stats_file, 'r') as fp:
        res = json.load(fp)
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
            plot_fig(mfr, y_mfr_sem, divs, "# spikes/s", b, e, "MFR", mfr_ch_num, image_folder)
            time.sleep(wait_time)
            #
            plot_fig(mbr, y_mbr_sem, divs, "# bursts/min", b, e, "MBR", mbr_ch_num, image_folder)
            time.sleep(wait_time)
            #
            plot_fig(mfib, y_mfib_sem, divs, "# spikes/burst", b, e, "MFIB",
                     mfib_ch_num, image_folder)
            time.sleep(wait_time)
            #
            plot_fig(mburdur, y_mburdur_sem, divs, "s", b, e, "MBURDUR",
                     mburdur_ch_num, image_folder)
            time.sleep(wait_time)
            #
            plot_fig(mnrs, y_mnrs_sem, divs, "% randoms spikes", b, e, "MNRS",
                     mnrs_ch_num, image_folder)
            time.sleep(wait_time)

def main():
    parser = argparse.ArgumentParser(description =
    '''The convert_sim_results.py script converts simulation output data into \
    the same forma used for storing the experimental spiking activity.''')
    parser.add_argument("--inputfolder", type=str, required=False,
        default="./output", help="folder containing the .json file to be \
        analysed")
    parser.add_argument("--configfile", type=str, required=False,
        default="config.json", help="configuration file containing the \
        parameters to be taken into account when computing stats; \
        default is config.json")
    parser.add_argument("--parsename", type=str, required=False, default="",
            help="only compute stats on .json  files whose names contain the \
            givenstring; default is ''")
    args = parser.parse_args()

    inputfolder = args.inputfolder
    configfile = args.configfile
    parsename = args.parsename



    # main
    with open(configfile, "r") as pf:
        config = json.load(pf)

    make_plots = config["plot_stats"]["make_plots"]

    ld = os.listdir(inputfolder)
    for l in ld:
        if l[-8:] == "npy.json" and (parsename == "" or parsename in l):
            data_file = os.path.join(inputfolder, l)
            stats_file =  data_file + ".stats.json"
            config_file =  data_file + "stats.config.json"

            with open(data_file, 'r') as fp:
                data = json.load(fp)

            all_stats = {
                "batches":{},
                "parameters": config["compute_stats"]
                }

            for batch in data.keys():
                print("###########")
                print("Batch " + batch)
                all_stats["batches"][batch] = {}
                for exp in data[batch].keys():
                    all_stats["batches"][batch][exp] = {}
                    print("Exp " + exp)
                    for div in data[batch][exp].keys():

                        div_int = int(div.lower().replace("div",""))
                        print(div_int)
                        if div_int > 60000:
                            continue

                        ast = all_stats["batches"][batch][exp][div] = {}
                        all_stats["batches"][batch][exp][div]["rec_len"] = \
                            data[batch][exp][div]["rec_len"]
                        #
                        all_stats["batches"][batch][exp][div]["mfr"] = {}
                        all_stats["batches"][batch][exp][div]["mfr"]["mean"] = 0
                        all_stats["batches"][batch][exp][div]["mfr"]["sem"] = 0
                        all_stats["batches"][batch][exp][div]["mfr"]["chs"] = {}

                        #
                        all_stats["batches"][batch][exp][div]["mbr"] = {}
                        all_stats["batches"][batch][exp][div]["mbr"]["mean"] = 0
                        all_stats["batches"][batch][exp][div]["mbr"]["sem"] = 0
                        all_stats["batches"][batch][exp][div]["mbr"]["chs"] = {}
                        #
                        all_stats["batches"][batch][exp][div]["mfib"] = {}
                        all_stats["batches"][batch][exp][div]["mfib"]["mean"] = 0
                        all_stats["batches"][batch][exp][div]["mfib"]["sem"] = 0
                        all_stats["batches"][batch][exp][div]["mfib"]["chs"] = {}
                        #
                        all_stats["batches"][batch][exp][div]["mnrs"] = {}
                        all_stats["batches"][batch][exp][div]["mnrs"]["mean"] = 0
                        all_stats["batches"][batch][exp][div]["mnrs"]["sem"] = 0
                        all_stats["batches"][batch][exp][div]["mnrs"]["chs"] = {}
                        #
                        all_stats["batches"][batch][exp][div]["mburdur"] = {}
                        all_stats["batches"][batch][exp][div]["mburdur"]["mean"] = 0
                        all_stats["batches"][batch][exp][div]["mburdur"]["sem"] = 0
                        all_stats["batches"][batch][exp][div]["mburdur"]["chs"] = {}

                        compute_stats(data[batch][exp][div], ast, config)

            with open(stats_file, 'w') as fp:
                json.dump(all_stats, fp)
            with open(config_file, 'w') as fp:
                json.dump(config, fp)



            if make_plots:
                image_folder = os.path.join(os.path.dirname(data_file),
                    os.path.basename(stats_file) + "plots")
                print(image_folder)
                plot_stats(stats_file, config, image_folder)




if __name__ == "__main__":
    main()
