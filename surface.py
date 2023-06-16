import argparse
import numpy as np
import pandas as pd
from salt_bridges import (
    data_coord_extraction,
    dist_calc,
)


def print_grid(p):
    print(len(p))
    for i in p:
        print("H\t", "\t".join(i.astype(str)), "\t1")


def surface_amino_acids(
    file_path: str,
    resolution: int | float = 3,
    threshold: int | float = 3.5,
    border: int | float = 4,
    outfile: str | None = None,
    sele_chain: str | None = None,
    silent: bool = True,
):
    """find amino acids on the surface of a protein
    :parameter
        - file_path:
          path to the input pdb file
        - resolution:
          spacing of the grid points
        - threshold:
          additional distance a grid point can have to a protein coordinate and count
          as protein point
        - border:
          how much the grid should be expanded over the furthest out protein coordinates
        - outfile:
          file path where the output file should be stored
        - sele_chain:
          ChainID if everything should be done for only one chain
        - silent:
          if no output should be printed

    :return
        - func1return
          description
    """
    # read pdb file data
    data, coords = data_coord_extraction(file_path)

    # use either whole data or only of the selected chain of sele_chain
    if sele_chain is not None:
        chain_test = data[:, 2] == sele_chain
    else:
        chain_test = np.ones(len(data)).astype(bool)
    data = data[chain_test]
    coords = coords[chain_test]

    d = {
        "atom": data[:, 0],
        "aminoAcid": data[:, 1],
        "chain": data[:, 2],
        "num": data[:, 3].astype(int),
    }
    df = pd.DataFrame(data=d)

    # generate axis point ranges
    z_p = np.arange(
        np.min(coords[:, 2]) - border,
        np.max(coords[:, 2]) + border,
        resolution,
    )
    y_p = np.arange(
        np.min(coords[:, 1]) - border,
        np.max(coords[:, 1]) + border,
        resolution,
    )
    x_p = np.arange(
        np.min(coords[:, 0]) - border,
        np.max(coords[:, 0]) + border,
        resolution,
    )

    # generate grid around protein with defined resolution
    points = np.stack(np.meshgrid(z_p, y_p, x_p, indexing="ij"), 3).reshape(-1, 3)[
        :, ::-1
    ]
    # calculate distances between all grid points and all protein points
    distances = dist_calc(coords, points)
    # grid points that can be flagged as protein points and its surrounding
    true_dist = (
        distances
        <= np.sqrt(resolution**2 + resolution**2 + resolution**2) + threshold
    )
    # grid points that are near to a protein point
    protein_points = np.any(true_dist, axis=0)

    # grid points that are not near enough to a protein point => SOLVENT
    solvent = distances >= resolution + threshold * 0.5
    solvent = np.all(solvent, axis=0)
    # grid points that are not to far away from the protein but also not
    # inside the protein => the SURFACE/SHELL
    shell_points = np.logical_and(solvent, protein_points)
    # which protein points are near the shell
    pos_surface_aa = true_dist * shell_points
    # which atom is near a solvent point
    surface_aa = np.any(pos_surface_aa, axis=1)
    df["surfaceAA"] = surface_aa
    # get the amino acid names/data if they are near the surface
    check = df.groupby(["aminoAcid", "chain", "num"], as_index=False)["surfaceAA"].any()
    check = check.sort_values(["chain", "num"])
    if outfile is not None:
        check.to_csv(outfile, index=False)
    if not silent:
        print(check[check["surfaceAA"]])


def arg_dict() -> dict:
    """argparser for salt bridges search
    :parameter
        - None:
    :return
        - d
          dictionary specifying all parameters for surface_amino_acids
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f", "--file_path", type=str, required=True, help="path to pdb file"
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=float,
        required=False,
        default=3.0,
        help="grid point spacing",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        required=False,
        default=3.5,
        help="additional distance a grid point can have to a protein coordinate and"
        "count as protein point",
    )
    parser.add_argument(
        "-b",
        "--border",
        type=float,
        required=False,
        default=4.0,
        help="how much the grid should be expanded over the furthest out protein"
        "coordinates",
    )
    parser.add_argument(
        "-c", "--create_file", type=str, required=False, default=None, help="file name"
    )
    parser.add_argument(
        "-s", "--not_silent", action="store_false", help="set flag to not show output"
    )
    parser.add_argument(
        "-se",
        "--sele_chain",
        type=str,
        required=False,
        default=None,
        help="ChainID if the hydrophobic cluster should "
        "only calculated for one specific chain",
    )
    args = parser.parse_args()
    d = {
        "file_path": args.file_path,
        "outfile": args.create_file,
        "resolution": args.resolution,
        "threshold": args.threshold,
        "outfile": args.create_file,
        "sele_chain": args.sele_chain,
        "silent": args.not_silent,
    }
    return d


if __name__ == "__main__":
    pass
    surface_amino_acids(**arg_dict())
