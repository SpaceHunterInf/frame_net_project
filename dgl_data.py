from dgl.data import DGLDataset
from torch.utils.data import Dataset
import dgl
import torch

class edsDataset(DGLDataset):

    def __init__(self, minibatch):
        self.graphs = []
        for d in minibatch:
            start = d['edge_index'][0]
            end = d['edge_index'][1]
            #print(start)
            #print(end)
            g = dgl.graph((start, end), num_nodes=d['x'].shape[0])
            g = dgl.add_self_loop(g)
            g.ndata['feat'] = d['x']
            g.ndata['mask'] = d['mask']
            g.ndata['label'] = torch.zeros((d['x'].shape[0], d['y'].shape[1])) - 1
            g.ndata['label'][d['mask']] = d['y'] #y is a one-hot tensor

            #print(g.ndata['label'].shape)
            self.graphs.append(g)
        self.num_features = self.graphs[0].ndata['feat'].shape[1]
        self.num_classes = self.graphs[0].ndata['label'].shape[1]

    def __getitem__(self, idx):
        """ Get graph and label by index
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.graphs[idx].ndata['label'].shape[1]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)


class edgeDataset(Dataset):
    def __init__(self, batch):
        self.data = []
        for d in batch:
            self.data.append({'feat':d['feat'], 'label':d['label'].flatten()})
        self.num_features = self.data[0]['feat'].shape[0]
        self.num_classes = self.data[0]['label'].shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)


    