import pandas as pd
import json
import numpy as np
N=400
input=["../../data/raw/abstracts_800.json"]
output="../../data/raw/doccano_400.jsonl"

def get_json(file_name):
    data = []
    for file in file_name:
        with open(file) as f:
            for line in f:
                row=json.loads(line)
                data.append(row)
    return data

dataset=get_json(input)
np.random.shuffle(dataset)
data=[{"text":l["paperAbstract"].replace("\n"," "),"labels":[]} for l in dataset[-N:]  if l["paperAbstract"].replace("\n","")!=""]

with open(output, 'a') as f:
    for d in data:
        f.write(json.dumps(d)+"\n")






