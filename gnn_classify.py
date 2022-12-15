import torch_geometric
from torch_geometric.data import Data
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from prepare_data import *

from sklearn.model_selection import train_test_split

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}

class GNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs,):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=in_channels,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)
        classification_out_channel = c_out
        self.linear = torch.nn.Linear(in_channels, classification_out_channel)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return {'before_classifier':x, 'after_classifier': self.linear(x)}

class Edge_MLP(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of hidden layers
            dp_rate - Dropout rate to apply throughout the network
        """
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        """
        Inputs:
            x - Input features per node
        """
        return self.layers(x)

class NodeLevelGNN(pl.LightningModule):

    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data):
        #Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
        
        x, edge_index = data.x.to(torch.float32), data.edge_index.to(torch.int64)
        mask = data.mask
        print(mask)
        y = data.y.to(torch.float32)
        x = self.model(x, edge_index)['after_classifier']

        loss = self.loss_module(x[mask], y)
        if x[mask].argmax(dim = -1) == y.argmax(dim = -1):
            acc = 1
        else:
            acc = 0
        return loss, acc, x[mask].argmax(dim = -1)

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc, _= self.forward(batch)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc, _ = self.forward(batch)
        self.log('val_acc', acc)


class EdgeLevelMLP(pl.LightningModule):
    def __init__(self, feature_config, GNN, **model_kwargs):
        self.feature_config = feature_config
        self.feature_config = feature_config

        self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()
        self.gnn_model = GNN

    def forward(self, x):
        node_features, edge_index, node2id = eds_encode(input['sentence'])
        edge_start_embedding = x[node2id[input['start']]]
        edge_end_embedding = x[node2id[input['end']]]
        edge_feature = torch.cat((edge_start_embedding, edge_end_embedding), 0)

        with torch.no_grad():
            x = self.gnn_model(node_features, edge_index.t().contigous)
        x = self.model(edge_feature)
        y = input[[target_fn_role_id]]
        if int(x.argmax(dim=-1)) == y:
            acc = 1
        else:
            acc = 0
        y = torch.nn.functional.one_hot(torch.tensor([y]), num_classes=len(x)).to(torch.float)
        
        loss = self.loss_module(x, y)
        return loss, acc

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch)
        self.log('test_acc', acc)


def train_node_classifier(model_name, dataset, num_labels, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = geom_data.DataLoader(dataset, batch_size=32)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join('node_level', "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)

    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu",
                         devices=1,
                         max_epochs=5,
                         enable_progress_bar=False) # False because epoch size is 1
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training

    pl.seed_everything()
    model = NodeLevelGNN(model_name=model_name, c_in=835, c_hidden=400, c_out=num_labels, **model_kwargs)
    trainer.fit(model, node_data_loader, node_data_loader)
    
    model.save_pretrained(root_dir)
    # Test best model on the test set
    # test_result = trainer.test(model, node_data_loader, verbose=False)
    # batch = next(iter(node_data_loader))
    # batch = batch.to(model.device)
    # _, train_acc = model.forward(batch, mode="train")
    # _, val_acc = model.forward(batch, mode="val")
    # result = {"train": train_acc,
    #           "val": val_acc,
    #           "test": test_result[0]['test_acc']}
    return model

def test_accuracy(model, data):
    correct = 0
    for x in data:
        with torch.no_grad():
            _,_,predict = model.forward(x)
        if predict == x.y:
            correct +=1
    
    return correct/len(data)


if __name__ == '__main__':
    with open('fn_frame2id.json','r') as f:
        frame2id = json.load(f)
    num_frame_label = len(frame2id.keys())
    with open('fn_role2id.json','r') as f:
        roles2id = json.load(f)
    num_role_label = len(roles2id.keys())

    with open('verb_transformed.pkl','rb') as f:
        verb_data = joblib.load(f)

    train_dataset = edsDataset(verb_data[:100])
    train_node_classifier('GNN', train_dataset, num_frame_label)
    #835, 336


    
    