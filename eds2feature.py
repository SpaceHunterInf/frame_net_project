import torch, json, joblib
from torch_geometric.data import Data
from prepare_data import *

if __name__ == '__main__':
    with open('verb_data.pkl', 'rb') as f:
        inputs = joblib.load(f)
    with open('fn_frame2id.json','r') as f:
        fn_frame2id = json.load(f)
    verb_encoded_data = []
    for input in inputs:
        node_features, edge_index, node2id = eds_encode(input['sentence'])
        mask = [False for node in input['eds'].nodes]
        mask[node2id[input['verb_id']]] = True
        y = torch.nn.functional.one_hot(torch.tensor([input['target_fn_frame_id']]), num_classes=len(fn_frame2id.keys())).to(torch.float)
        data = Data(x = node_features, edge_index = edge_index.t().contigous(), mask=mask, y=y)
        verb_encoded_data.append(data)
    
    with open('verb_transformed.pkl', 'wb') as f:
        joblib.dump(verb_encoded_data, f)