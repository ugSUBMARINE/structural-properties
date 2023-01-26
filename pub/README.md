This directory contains code to reproduce the results obtained for the publication and the data needed to do so.
It also can be used to use the same kind of analysis for a different set of proteins.

In order to install the required packages run:
```
python3 -m venv /PATH/TO/YOUR/VENV
source venv/bin/activate
pip install -r requirements.txt
```

## RMSF analysis
This is a script to compare (and visualize) the RMSF values of different proteins to a certain target value per protein and test their correlation.

All described settings can be changed/set in the `PARAMETERS` section in the `run_rmsf_analysis.py` file.
The structures of all proteins need to be specified in `structure_path`. The RMSF files are expected to be in a directory like `DIR_NAME/PROTEIN_NAME/FileNameWithIncreasingNumber.dat`. The multiple sequence alignment file (clustal w format) with the same sequences as the protein structures in the pdb files needs to be set in `algn_file`. 

Then the analysis can be run with `python3 run_rmsf_analysis.py`


## Property analysis
This contains the program protstrucprop. It can be used to calculate the hydrophobic cluster, H-bond networks and salt bridge networks of several structures using `properties`. Furthermore `greedy`and `directed` can be used to find the attribute combination to minimize the regression error for a certain target value and `predict` can be used to use a fitted regression model to predict the target value of interest for other proteins.

For general help of the programs use `python3 protstrucprop.py -h`

Supported models are:
* Linear Regression
* Ridge regression
* Random Forest Regression
* Gradient Boost Regression
* K-nearest neighbors Regression

To perform the analysis, the following steps must be carried out:
*   Calculating the properties for each protein 
    *`python3 protstrucprop.py properties --out_dir /PATH/TO/OUTPUT/DIR --struct_dir /PATH/TO/PDB/FILES --name_path /PATH/TO/PROTEIN/DATA`
    * The name/path to the directory containing all protein structures need to be set in `struct_dir` and the names of the files (without the .pdb) need to be set in `name_path`
    * `name_path`needs to be a tsv file where each row is a protein name like their pdb files (without .pdb) are named and a target value for each protein used in the regression later
    * This will calculate H-Bonds, hydrophobic cluster and salt bridges and store them in `out_dir`
    * Run `p3 protstrucprop.py properties -h` for parameter description


* Greedy search for the best attribute combination to minimize the prediction error
    * fits a linear regression model to the 3 most correlating attributes of each attribute group 
        * `python3 protstrucprop.py greedy --name_path /PATH/TO/PROTEIN/DATA --prop_dir /PATH/TO/OUTPUT/DIR --param_number 3 -regressor LR`
        * tests each possible combination starting with single to `param_number` * 3 combinations 
        * will take long if `param_number` > 3 due to the high number of possible combinations
    * fits a ridge regression model to all possible 2 attribute combinations
        * `python3 protstrucprop.py greedy --name_path /PATH/TO/PROTEIN/DATA --prop_dir /PATH/TO/OUTPUT/DIR --explore_all --force_nparam 2 --regressor RI `
        * a `force_nparam` > 5 will take long due to the high number of possible combinations
    * searches for the best attribute combination given a certain search method
        * `python3 protstrucprop.py directed --name_path /PATH/TO/PROTEIN/DATA --prop_dir /PATH/TO/OUTPUT/DIR --search_method 1 -regressor KNN --silent`
            * forward search - greedy checks attributes and adds the attribute that creates the lowest error to produce the same check in the next round for attributes that have not been added so far and returns the best found combination
            * backward search - reverse version of forward seach - removes one attribute per round whos removal leads to the lowest error
            * model based search - same as backward search but removes one attribute per round that has the lowest feature importance (not possible for k-nearest neighbors)
            * `silent` supresses the biggest outputs in the terminal
    * Run `p3 protstrucprop.py greedy -h` and `p3 protstrucprop.py directed -h` for parameter description
        * cross validation mode can be changed
        * search can be run in parallel
        * search course plots and feature importance plots can be made
        * models can be saved
* Predict with fitted models
    * predicts values for proteins with calculate attributes
        * `python3 protstrucprop.py predict --name_path /PATH/TO/PROTEIN/DATA --model_name NAMEOFTHEMODEL --prop_dir /PATH/TO/OUTPUT/DIR
        * `name_path` is a tsv file like used above but can have an empty colum for target values
    * Run `p3 protstrucprop.py predict -h` for parameter description


### File structures from property analysis
File structure created by `protstrucprop properties`
```
'- out_dir
    |- PROTEIN_NAME_hb.csv
    |- PROTEIN_NAME_hy.csv
    '- PROTEIN_NAME_sb.csv
```
File structure created by `protstrucprop greedy` and `protstrucprop directed`
```
'- saved_models
    |- MODELNAME.save
    |- MODELNAME_scaler.save
    '- MODELNAME_settings.txt
```
File structure created by `protstrucprop predict`
```
'- results
    '- MODELNAME.json
```


