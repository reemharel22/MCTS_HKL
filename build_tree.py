import json
import sys

root = []
with open(sys.argv[1]) as json_file:
    data = json.load(json_file)['nodes']
    root = data[0]
    root['children'] = []
    data[0]['node'] = root
    for i in range(1,len(data)):
        data[i]['children'] = []
        data[data[i]['parent']-1]['children'].append(data[i])
        data[i]['node'] = data[data[i]['parent']-1]['children'][-1]

    for node in data:
        del node['node']
        del node['parent']

    with open(sys.argv[1]+'parsed.json','w') as json_out_file:
        json.dump(root,json_out_file)