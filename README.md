![TEST](https://github.com/ugSUBMARINE/structural-properties/actions/workflows/test.yml/badge.svg)

This is a repository in which 3 stand-alone programmes for the calculation of hydrogen bond networks, salt bridge networks and hydrophobic clusters inside a protein structure are available.
To calculate these properties, a pdb file of the protein of interest is needed.
Only entries in the pdb file with marked as ATOM are considered. Duplicated side chain entries are not allowed. It is possible to select one chain or to use all chains in the pdb file for the calculation.
Each of the files contains an argparser and can be used with the filepath to the pdb file of the protein of interest as only argument.
They calculate the specific interactions and build clusters of interacting side chains. If desired, these are also saved in a file.


**Software Requirements:**
*  [Python3.10](https://www.python.org/downloads/)

*optional:*
*  [Anaconda or Miniconda](https://docs.anaconda.com/anaconda/install/index.html)

To execute please run one of the following commands:

`python3 h_bond_test.py -f /PATH/TO/PDB/FILE`

`python3 hydrophobic_cluster.py -f /PATH/TO/PDB/FILE`

`python3 salt_bridges.py -f /PATH/TO/PDB/FILE`

For more details to the specific argument please run `python3 XXX.py -h` 

The `pub` directory contains the scripts and data that was used to perform all the structure related analyses in the paper ""
