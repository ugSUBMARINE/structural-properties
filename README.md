This is a repository in which 3 stand-alone programmes for the calculation of hydrogen bonds, salt bridges and hydrophobic clusters are available. Each of them contains an argparser and can be used with the filepath to the pdb file of the protein of interest as only argument.
It calculates the specific interactions and builds clusters of interacting side chains. If desired, these are also saved in a file.

To execute please run one of the following commands:

`python3 h_bond_test.py -f /PATH/TO/PDB/FILE`

`python3 hydrophobic_cluster.py -f /PATH/TO/PDB/FILE`

`python3 salt_bridges.py -f /PATH/TO/PDB/FILE`

For more details to the specific argument please run `python3 XXX.py -h` 
