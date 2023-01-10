This directory contains code to reproduce the results obtained for the publication and the data needed to do so.
It also can be used to use the same kind of analysis for a different set of proteins.

## RMSF analysis
All described settings can be changed/set in the `PARAMETERS` section in the `run_rmsf_analysis.py` file.
The structures of all proteins need to be specified in `structure_path`. The RMSF files are expected to be in a directory like `DIR_NAME/PROTEIN_NAME/FileNameWithIncreasingNumber.dat`. The multiple sequence alignment file (clustal w format) with the same sequences as the protein structures in the pdb files needs to be set in `algn_file`. 

Then the analysis can be run with `python3 run_rmsf_analysis.py`


## Property analysis
To perform the analysis, the following steps must be carried out:
*   Calculating the data for each protein with `python3 calc_prop.py`
    * uses function `calculate`
    * all .pdb files need to be stored in one directory
    * In this file the name/path to the directory containing all protein structures need to be set in `struct_dir` and the names of the files (without the .pdb) need to be set in `p_names`. 
    * This will calculate H-Bonds, hydrophobic cluster and salt bridges and store them in `out_dir`.
* run `python3 run_property_analysis.py`
    * uses function `fit_data`
    * Set the `data_path` to the same as `out_dir` chosen in `calc_prop`
    * Define the model name in `save_model` using a desired name instead of None
    * Set `model_ind` to either `None` to use the model with the best (lowest) mean squared error, e.g. `1` for the model with index 1 or e.g. `"!4"` for the best model of the best 10 that is closest to having 4 parameters to store the model and also it's used attributes from `*_DATA_DESC`. 
    * use `force_mi` to only store a desired model if it has the desired number of parameters set with `!`
    * change `target` if data should be fit to something else than `temp_cd`
    * add proteins and temps with `add_names` and `add_temp` respectively
    * set number of attributes with `*_param_num`
    * overwrite the best correlating attributes with `ow_*_vals`
    * changed protein names from global `p_names` with `p_names_in`
    * use `force_np` not test all combinations from 1 to 9 but only the number of combinations set
    * set `explore_all` to ignore `*_vals` and use all `*_DATA_DESC` to find best parameter combination
* run `python3 predict.py`
    * uses function `predict`
    * Set the `model_name` like specified in `save_model` from `run_property_analysis.py` and the `data_dir` like `out_dir` from `calc_prop.py`.
    * add proteins that are not in the `p_names` put in the `out_dir` to calculate their values using a given fitted model in `add`
    * set `save_results` to save all predictions for all proteins as json.
    * This will predict the desired value and the ordering of all proteins


For all files, if `p_names` are different than the set on add `p_names = np.array(["Protein1", "Protein2"])`.

### File structures from property analysis
File structure created by `calc_prop.py`
```
'- out_dir
    |- PROTEIN_NAME_hb.csv
    |- PROTEIN_NAME_hy.csv
    '- PROTEIN_NAME_sb.csv
```


