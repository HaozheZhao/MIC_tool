

import argparse
from pathlib import Path
from hydra import compose, initialize

import common.io_utils as ioutils
import common.lib_utils as libutils
from taskgen.collect import collect


def parse_args():
    parser = argparse.ArgumentParser(description="VL-taskgen source")
    parser.add_argument("--cfg", default="prompts/image_captioning/ln_coco-prompt-2/configs.yaml", type=str,
                        help="The configuration file.")
    args = parser.parse_args()
    return args


def main(args): 
    config_file = Path(args.cfg)
    try:
        with initialize(version_base=None, config_path=str(config_file.parent)):
            cfg = compose(config_name=config_file.name)
    except:
        raise NotImplementedError(f"No configuration file at {args.cfg}.")
    output_dir = Path(cfg.output.path)
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)

    all_subtasks = collect(cfg)
    split_type = ['train','val','test']
    for subset_idx, subset in enumerate(all_subtasks): 
        task_id = libutils.fetch_task_id(output_dir)
        ioutils.save_json(subset, output_dir / f"task{task_id}-{cfg.task.name}-{cfg.dataset.name}-prompt-{cfg.dataset.prompt}-subset-{split_type[subset_idx]}.json")


if __name__ == "__main__":
    args = parse_args()
    main(args)