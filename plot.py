import matplotlib.pyplot as plt
import numpy as np
import json
from utils import get_config
import glob 

CONFIG_PATH = 'config.json'
METRICS =  ["loss", "precision", "recall", "f1", "accuracy"]
config = get_config(CONFIG_PATH)


def plot_files(param, param_files, results):
    for metric in METRICS:
        for file_name, res in zip(param_files, results):
            metric_vals = [round(epoch_res[metric],5) for epoch_res in res]
            name = file_name.split('\\')[1].replace('.json', '')
            plt.plot(np.arange(0,config.epochs), metric_vals, label=name)

        plt.title(f'{param.replace("_", " ")} {metric}'.title())
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(f'plots/{param}_{metric}.png')
        plt.cla()
        plt.clf()
        plt.show()

res_files = glob.glob(config.save_metric_dir + '*') 

for param, values in config.tested_params.items():
    param_files = [name for name in res_files if param in name]
    results = [json.load(open(name, 'r')) for name in param_files if param in name]
    plot_files(param, param_files, results)

