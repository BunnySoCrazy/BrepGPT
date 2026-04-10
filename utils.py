import os
import json
import torch

specifications_filename = "specs.json"


def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_experiment_specifications(experiment_directory):
    print("loading specifications of " + experiment_directory)
    filename = os.path.join(experiment_directory, specifications_filename)
    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )
    return json.load(open(filename))
