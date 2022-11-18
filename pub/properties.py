import os
import sys
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt


class SaltBridges:
    def __init__(self, detailed_filepath, server_filepath, data_description):
        self.detailed_filepath = detailed_filepath
        self.server_filepath = server_filepath
        self.data_description = data_description

    def read_detailed_data(self):
        data = pd.read_csv(self.detailed_filepath, delimiter=",")
        return (
            np.asarray(data["InteractingResidues"]),
            np.asarray(data["ContactsPerCluster"]),
        )

    def read_server_data(self):
        data = pd.read_csv(self.server_filepath, delimiter=",")
        data_description = ["FCR", "Kappa"]
        return (
            np.asarray(data["protein"]),
            np.asarray(data["FCR"]),
            np.asarray(data["Kappa"]),
            data_description,
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
            self.data_description,
        )


class HydrophobicClusterOwn:
    def __init__(self, filepath, data_description):
        self.filepath = filepath
        self.data_description = data_description

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
            self.data_description,
        )



if __name__ == "__main__":
    pass
