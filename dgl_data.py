from dgl.data import DGLDataset
import dgl

class edsDataset(DGLDataset):

    def __init__(self, minibatch):
        self.graphs = []
        for d in minibatch:
            start = d['edge_index'][0]
            end = d['edge_index'][1]
            g = dgl.graph((start, end), num_nodes=d['x'].shape[0])
            g.ndata['feat'] = d['x']
            g.ndata['mask'] = d['mask']
            g.ndata['label'] = d['y'] #y is one-hot vector
            self.graphs.append(g)
        self.num_features = self.graphs[0].ndata['feat'].shape[1]
        self.num_classes = self.graphs[0].ndata['y'].shape[0]

    def __getitem__(self, idx):
        """ Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.graphs[idx].ndata['y']

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)