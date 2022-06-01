import scikit_posthocs as sp
import numpy as np
import json
import glob
from utils import get_config
from collections import defaultdict

CONFIG_PATH = 'config.json'
config = get_config(CONFIG_PATH)

def convert_to_lists(res_dict):
    res = defaultdict(list)

    for sub in res_dict:
        for key in sub:
            res[key].append(sub[key])

    return dict(res)

with open('stats_results.txt', 'w+', encoding='UTF-8') as results_file:

    for param, values in config.tested_params.items():
        results = {}
        for value in values:
            with open(config.save_metric_dir + f"{param}_{value}.json", "r") as res_file:
                results[f"{param}_{value}"] = convert_to_lists(json.load(res_file))
        
        aggregated_results = {
            "loss": [],
            "recall": [],
            "precision": [],
            "f1": [],
            "accuracy": [],
        }
          
        for name, metrics in results.items():
            aggregated_results["loss"].append(results[name]['loss'])
            aggregated_results["recall"].append(results[name]['recall'])
            aggregated_results["precision"].append(results[name]['precision'])
            aggregated_results["f1"].append(results[name]['f1'])
            aggregated_results["accuracy"].append(results[name]['accuracy'])
        
        for key, aggregated_result in aggregated_results.items():
            results_file.write(f"Analiza {param} - {key}: \n")
            data = np.array([*aggregated_result])
            res = sp.posthoc_nemenyi_friedman(data.T)
            print(res)
            results_file.write(res.to_string())
            results_file.write('\n\n')
        
        results_file.write('\n\n\n')
    

