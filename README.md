![TEST](https://github.com/ugSUBMARINE/structural-properties/actions/workflows/test.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is a repository in which 3 stand-alone programmes for the calculation of hydrogen bond networks, salt bridge networks and hydrophobic clusters inside a protein structure are available.
To calculate these properties, a pdb file of the protein of interest is needed.
Only entries in the pdb file with marked as ATOM are considered. Duplicated side chain entries are not allowed. It is possible to select one chain or to use all chains in the pdb file for the calculation.
Each of the files contains an argparser and can be used with the filepath to the pdb file of the protein of interest as only argument.
They calculate the specific interactions and build clusters of interacting side chains. If desired, these are also saved in a file.

In the `pub` directory, one finds `protstructprop` which can be used to perform these calculations for a given set of proteins and also to fit different regression models to data provided with the set of proteins (Fitting the thermal stability to a set of proteins and predict it for not tested proteins for example).


**Software Requirements:**
*  [Python3.10](https://www.python.org/downloads/)

*optional:*
*  [Anaconda or Miniconda](https://docs.anaconda.com/anaconda/install/index.html)

In order to install the required packages run:
```
python3 -m venv /PATH/TO/YOUR/VENV
source venv/bin/activate
pip install -r requirements.txt
```

To execute please run one of the following commands:

`python3 h_bond_test.py -f /PATH/TO/PDB/FILE`

`python3 hydrophobic_cluster.py -f /PATH/TO/PDB/FILE`

`python3 salt_bridges.py -f /PATH/TO/PDB/FILE`

For more details to the specific argument please run `python3 XXX.py -h` 

The `pub` directory contains the scripts and data that was used to perform all the structure related analyses in the paper ""
