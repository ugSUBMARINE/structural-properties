This directory contains code to reproduce the results obtained for the publication and the data needed to do so.
It also can be used to use the same kind of analysis for a different set of proteins.

## RMSF analysis
All described settings can be changed/set in the `PARAMETERS` section in the `run_rmsf_analysis.py` file.
The structures of all proteins need to be specified in `structure_path`. The RMSF files are expected to be in a directory like `DIR_NAME/PROTEIN_NAME/FileNameWithIncreasingNumber.dat`. The multiple sequence alignment file (clustal w format) with the same sequences as the protein structures in the pdb files needs to be set in `algn_file`. 

Then the analysis can be run with `python3 run_rmsf_analysis.py`


## Property analysis
In order to run the property analysis different steps need to be done:
*   Calculating the data for each protein with `python3 calc_prop.py`
    * In this file the name/path to the directory containing all protein structures need to be set in `struct_pub` and the names of the files (without the .pdb) need to be set in `p_names`. 
    * This will calculate H-Bonds, hydrophobic cluster and salt bridges and store them in `parent_name` which has also to be set in the files `PARAMETERS` section.
* run `python3 run_property_analysis.py`
    * The first time one just needs to set the `data_path` to the same as `parent_name` chosen in `calc_prop`. 
    * Checking the single correlations and choosing the highest correlating ones yields to the best results.
        * The indices of the highest correlating attributes need to be set in `property_analysis.py` in `HB_SELE` for H-Bonds, in `SB_SELE` for salt bridges and in `HY_SELE` for hydrophobic cluster. 
        * The models with the best parameters can be seen and the index can be used to test and/or store this model (normally done in the next step)
    * Run `python3 run_property_analysis.py` again but in addition to the selected attributes, change `save_model` from None to a desired name and `model_ind` to either None to use the model with the best (lowest) mean squared error, e.g. `1` for the model with index 1 or e.g. "!4" for the best model of the best 10 that is closest to having 4 parameters to store the model and also it's used parameters from `HB_SELE`, `SB_SELE` and `HY_SELE`.
* run `python3 predict.py`
    * Set the `model_name` like specified in `save_model` from `run_property_analysis.py` and the `data_dir` like `parent_name` from `calc_prop.py`.
    * This will predict the desired value and the ordering of all proteins

For all files, if `p_names` are different than the set on add `p_names = np.array(["Protein1", "Protein2"])` in the `PARAMETERS` section.
