import pandas as pd
import json
import numpy as np
N=200
input=["../abstracts.json"]
output="../output.json"


def get_json(file_name):
    print (file_name)
    data = []
    for file in file_name:
        with open(file) as f:
            for line in f:
                #print (line)
                row=json.loads(line)
                data.append(row)
    return data

data=[]
for l in get_json(input)[-N:]:
    data.append({"text":l["text"].encode("utf-8"),"labels":[]} )

print (data)
#pd.DataFrame(data).reset_index().to_json(output)
#np.savetxt(output, data)
with open(output, 'a') as f:
    for d in data:
        f.write( json.dumps(d)+"\n")






