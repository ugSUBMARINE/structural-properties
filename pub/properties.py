import numpy as np
import pandas as pd


charge = {
    "ALA": 0,
    "CYS": 0,
    "ASP": -1,
    "GLU": -1,
    "PHE": 0,
    "GLY": 0,
    "HIS": 1,
    "ILE": 0,
    "LYS": 1,
    "LEU": 0,
    "MET": 0,
    "ASN": 0,
    "PRO": 0,
    "GLN": 0,
    "ARG": 1,
    "SER": 0,
    "THR": 0,
    "VAL": 0,
    "TRP": 0,
    "TYR": 0,
}

hydropathy = {
    "ALA": 1.8,
    "CYS": 2.5,
    "ASP": -3.5,
    "GLU": -3.5,
    "PHE": 2.8,
    "GLY": -0.4,
    "HIS": -3.2,
    "ILE": 4.5,
    "LYS": -3.9,
    "LEU": 3.8,
    "MET": 1.9,
    "ASN": -3.5,
    "PRO": -1.6,
    "GLN": -3.5,
    "ARG": -4.5,
    "SER": -0.8,
    "THR": -0.7,
    "VAL": 4.2,
    "TRP": -0.9,
    "TYR": -1.3,
}


class SaltBridges:
    def __init__(self, detailed_filepath):
        self.detailed_filepath = detailed_filepath

    def read_detailed_data(self):
        data = pd.read_csv(self.detailed_filepath, delimiter=",")
        return (
            np.asarray(data["InteractingResidues"]),
            np.asarray(data["ContactsPerCluster"]),
        )

    def network_data(self):
        i_res, interactions = self.read_detailed_data()
        # number of salt bridges between chains
        interchain = 0
        # number of residues in each cluster
        cluster_size = []
        for i in i_res:
            i_split = i.split(" - ")
            cluster_size.append(len(i_split))
            chains_int = []
            for k in i_split:
                chains_int.append(k.split("-")[1])
            if np.unique(chains_int).shape[0] > 1:
                interchain += 1
        return interchain, cluster_size, interactions, i_res

    def stats(self):
        ic, cs, ia, _ = self.network_data()

        return (
            np.max(ia),
            np.min(ia),
            np.mean(ia),
            np.sum(ia),
            np.max(cs),
            np.min(cs),
            np.mean(cs),
            np.sum(cs),
            ic,
        )


class HydrophobicClusterOwn:
    def __init__(self, filepath):
        self.filepath = filepath

    def read_detailed_data(self):
        data = pd.read_csv(self.filepath, delimiter=",")
        return (
            np.asarray(data["InteractingResidues"]),
            np.asarray(data["ContactsPerCluster"]),
            np.asarray(data["SurfaceAreaPerCluster"]),
        )

    def network_data(self):
        i_res, interactions, surface_area_pc = self.read_detailed_data()
        # number of salt bridges between chains
        interchain = 0
        # number of residues in each cluster
        cluster_size = []
        for i in i_res:
            i_split = i.split(" - ")
            cluster_size.append(len(i_split))
            chains_int = []
            for k in i_split:
                chains_int.append(k.split("-")[1])
            if np.unique(chains_int).shape[0] > 1:
                interchain += 1
        return interchain, cluster_size, interactions, i_res, surface_area_pc

    def stats(self):
        ic, cs, ia, _, sapc = self.network_data()

        return (
            np.max(sapc),
            np.min(sapc),
            np.mean(sapc),
            np.sum(sapc),
            np.max(ia),
            np.min(ia),
            np.mean(ia),
            np.sum(ia),
            ic,
        )


class SurfaceProp:
    def __init__(self, filepath):
        self.filepath = filepath

    def stats(self):
        data = pd.read_csv(self.filepath, delimiter=",")
        aa_surf = data["aminoAcid"][data["surfaceAA"]].values
        charge_transform = np.asarray(list(map(charge.get, aa_surf)))
        hydropathy_transform = np.asarray(list(map(hydropathy.get, aa_surf)))
        return (
            np.sum(charge_transform != 0),
            np.sum(charge_transform),
            np.sum(hydropathy_transform),
        )


if __name__ == "__main__":
    a = SurfaceProp(
        "/home/gwirn/PhDProjects/ancestors/ancestor_pub/surface_test.csv"
    )
    print(a.stats())
    pass
