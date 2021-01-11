import os
import json
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import time


wait_time = 0
with open("data.json", 'r') as fp:
    data = json.load(fp)

# plot windows (couples of values) in s
plot_bin = [60, 90, 600, 630, 1080, 1110]

# create final folder 
image_folder = "raster_plots"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

for b in data.keys():
    for e in data[b].keys():
        
        # file name
        filename = b + "_" + e

        # sort divs
        divs = []
        divs_temp = [int(dd[3:]) for dd in data[b][e].keys()]
        sort_index = np.argsort(np.array(divs_temp))
        for i in sort_index:
            if len(str(divs_temp[i])) == 1:
                divs.append("div0" + str(divs_temp[i]))
            else:
                divs.append("div" + str(divs_temp[i]))
        #divs = ["div" + str(divs[i]) for i in sort_index]

        # create subplot titles
        subplot_title = []

        for k in divs:
            subplot_title.append(b + " - " + e + " - " +k)
            subplot_title.append(b + " - " + e + " - " +k)
            subplot_title.append(b + " - " + e + " - " +k)
        subplot_titles = tuple(subplot_title)

        # create figure
        fig = make_subplots(
            rows=len(divs), cols=3, 
            subplot_titles = subplot_titles
        )
        marker_size = 2
        color = "black"

        # for each div
        for didx, d in enumerate(divs):
            print(b, " ", e, " ",  d)

            # for each channel extract data to be plotted
            for cidx, c in enumerate(data[b][e][d]["chs"].keys()):
                ch_times = data[b][e][d]["chs"][c]
                ttimes = [0, 0, 0, 0, 0, 0]
                for ccounter, cel in enumerate(ch_times):
                    for ti_idx, ti in enumerate(ttimes):
                        if ti == 0 and cel > plot_bin[ti_idx]:
                            ttimes[ti_idx] = ccounter
                for ttt in range(len(ttimes)):
                    if ttimes[ttt] == 0:
                        ttimes[ttt] = len(ch_times)

                x = ch_times[ttimes[0]:ttimes[1]]
                y = len(x) * [int(c)]
                fig.add_trace(
                    go.Scatter(x=x, y=y, mode="markers", 
                    marker=dict(size=marker_size, color=color)
                    ),
                    row=didx+1, col=1)

                x = ch_times[ttimes[2]:ttimes[3]]
                y = len(x) * [int(c)]
                fig.add_trace(
                    go.Scatter(x=x, y=y, mode="markers",
                    marker=dict(size=marker_size, color=color)
                    ),
                    row=didx+1, col=2)
                
                x = ch_times[ttimes[4]:ttimes[5]]
                y = len(x) * [int(c)]
                fig.add_trace(
                    go.Scatter(x=x, y=y, mode="markers",
                    marker=dict(size=marker_size, color=color)
                    ),
                    row=didx+1, col=3)

        fig.update_layout(height=6000, showlegend=False)
        fig.write_html(os.path.join(image_folder, filename + ".html"))
