1) All the data, except for 
"batch20080430" -> "2008051202" -> "div33": [3,4,5,6]
were extracted from full length  ptrains (i.e. previously concatenated 5-min
chunks) contained in folder:
/Users/lbologna/MyWork/2005-2009_phd/data_backup_temp/Data/all_exp/ExpXpapModelFull/
with the following script:

python mat_to_json.py

---
2) the data referring to the missing div:
"batch20080430" -> "2008051202" -> "div33": [3,4,5,6]
were extracted by concatenating chunks [3,4,5,6] from the following folder:
/Users/lbologna/MyWork/2005-2009_phd/data_backup_temp/Data/all_exp/ExpXpapModel/
with the following script:

python mat_to_json_multi_chunks.py

---
3) the two data files obtained in steps 1) and 2) were merged with the 
following script:

merge_data.py

---
4) the analysis was run with the following script:

compute_stats.py

---
5) activity plots were obtained, from the results file, with the following script:

plot_stats.py


---
6) raster plots were obtained, from the data file, with the following script:

raster_plot.py

-------------

Point from 1) to 5) must be executed in order (and point 6) must be executed
after point 1) to 3) have been executed)

Names of input and output files/folder and parameters (e.g. for plotting) must
be set inside the python scripts.
In a future version of the code they will 
be passed as parameters.





