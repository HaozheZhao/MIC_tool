import csv
import json
import yaml
from pathlib import Path

from omegaconf import OmegaConf


def load_json(filename):
    with Path(filename).open("rb") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=True, sort_keys=False):
    with Path(filename).open("w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)

def load_jsonl(filename):
    with Path(filename).open("r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    with Path(filename).open("w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def load_yaml(filename):
    with Path(filename).open("r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def save_yaml(data, filename):
    with Path(filename).open("w") as f:
        yaml.dump(data, f, default_flow_style=False)


def load_csv(filename, delimiter=","):
    idx2key = None
    contents = {}
    with Path(filename).open("r") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for l_idx, row in reader:
            if l_idx == 0:
                idx2key = row
                for k_idx, key in enumerate(idx2key):
                    contents[key] = []
            else:
                for c_idx, col in enumerate(row):
                    contents[idx2key[c_idx]].append(col)
    return contents, idx2key


def save_csv(data, filename, cols=None, delimiter=","):
    with Path(filename).open("w") as f:
        writer = csv.writer(f, delimiter=delimiter)
        num_entries = len(data[list(data.keys())[0]])
        assert cols is not None, "Must have column names for dumping csv files."
        writer.writerow(cols)
        for l_idx in range(num_entries):
            row = [data[key][l_idx] for key in cols]
            writer.writerow(row)


def cfg2obj(cfg):
    cfg_dict = OmegaConf.to_container(cfg)
    return cfg_dict