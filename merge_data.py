import json

source_all = "data_all_except_one_div.json"
source_add = "data_reconstruct_one_div.json"
dest_all = "data_all.json"

with open(source_all, 'r') as f:
    dataf = json.load(f)


with open(source_add, 'r') as ff:
    dataff = json.load(ff)

dataf["batch20080430"]["2008051202"]["div33"] = \
        dataff["batch20080430"]["2008051202"]["div33"]


with open(dest_all, 'w') as fd:
    json.dump(dataf, fd)

