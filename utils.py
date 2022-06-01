import json

class Params:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_config(path):
    with open(path) as f:
        config = json.load(f)
        return Params(**config)

def save_results(path, results):
    with open(path,'w+') as res:
        json.dump(results, res)