import csv
import os
import re
from typing import Any, Dict, List, Tuple


class Experiment:
    def __init__(
        self, layer_size: int, num_layers: int, rank: int, data: Dict[str, int]
    ):
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.rank = rank
        self.data = data

    def add_data(self, name: str, mean: int):
        self.data[name] = mean

    def __str__(self):
        return f"Experiment(layer_size={self.layer_size}, num_layers={self.num_layers}, rank={self.rank}, data={self.data})"


def parse_filename(filename: str) -> Tuple[int, int, int]:
    """Extract layer_size, num_layers, and rank from filename."""

    pattern = r"_ls(\d+)_nl(\d+)_rank(\d+)\.csv"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    return tuple(map(int, match.groups()))


def read_csv_data(filepath: str) -> Dict[str, int]:
    """Read CSV file and extract name and mean columns."""

    data = {}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["name"]] = int(float(row["mean"]))
    return data


def process_folder(folder_path: str) -> List[Experiment]:
    """Process all CSV files in a folder and return list of Experiment objects."""

    experiments = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue

        filepath = os.path.join(folder_path, filename)
        layer_size, num_layers, rank = parse_filename(filename)
        data = read_csv_data(filepath)

        exp = Experiment(layer_size, num_layers, rank, data)
        experiments.append(exp)

    experiments.sort(key=lambda exp: (exp.layer_size, exp.num_layers, exp.rank))
    return experiments


def compare_experiments(mode_to_experiments: Dict[str, List[Experiment]]):
    modes = ["ray/no_overlap", "torch/fsdp"]
    print(f"# {modes[0]} {modes[1]}")
    print()

    exps1 = mode_to_experiments[modes[0]]
    exps2 = mode_to_experiments[modes[1]]
    cmp_keys = [
        ("total", "total"),
        ("fw.total", "fw.total"),
        ("bw.backward", "bw.bw_ar"),
        ("bw.others", "bw.bw_ar"),
        ("bw.update", "bw.update"),
    ]

    def print_diff(key1, key2, data1, data2):
        diff = data1 - data2
        rel = round(diff / data2 * 100)
        print(f"{key1} {key2} {data1} {data2} {diff} {rel}%")

    for i in range(len(exps1)):
        exp1 = exps1[i]
        exp2 = exps2[i]
        if exp1.rank != 0:
            continue
        print(f"{exp1.layer_size} {exp1.num_layers}")
        for key1, key2 in cmp_keys:
            assert key1 in exp1.data
            assert key2 in exp2.data
            data1 = exp1.data[key1]
            data2 = exp2.data[key2]
            print_diff(key1, key2, data1, data2)

        total2 = exp2.data["total"]

        total1 = (
            exp1.data["total"]
            - exp1.data["fw.total"]
            + exp2.data["fw.total"]
            - exp1.data["bw.update"]
            + exp2.data["bw.update"]
        )
        print_diff("total(+)fw_total(+)bw_update", "total", total1, total2)

        total1 = exp1.data["total"] - exp1.data["bw.others"]
        print_diff("total-bw_others", "total", total1, total2)

        print()


def main():
    args: Dict[str, Any] = {}
    args["folders"] = [
        "results/barbell/linear/ray/no_overlap/exp_self",
        "results/barbell/linear/torch/fsdp/exp_self",
    ]

    mode_to_experiments: Dict[str, List[Experiment]] = {}
    num_files = None

    for folder in args["folders"]:
        experiments = process_folder(folder)
        mode = folder.split("/")[-3] + "/" + folder.split("/")[-2]
        mode_to_experiments[mode] = experiments

        if num_files is None:
            num_files = len(experiments)
        elif num_files != len(experiments):
            raise ValueError(
                f"Inconsistent number of CSV files: {folder} has {len(experiments)} files, "
                f"expected {num_files}"
            )

    compare_experiments(mode_to_experiments)


if __name__ == "__main__":
    main()
