from dgl.nn import GraphConv
import dgl
import joblib
import torch
from torch import optim, nn, utils, Tensor
import torch.nn as nn
import torch.nn.functional as F
from dgl_data import *
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from tqdm import tqdm
import os, sys
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, h_feats)
        self.linear = torch.nn.Linear(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        y = self.linear(h)
        return h, y

class NodeGNN(pl.LightningModule):
    def __init__(self, in_feats, h_feats, num_classes, load=False, load_path=None):
        super().__init__()
        self.model = GCN(in_feats, h_feats, num_classes)
        if load:
            self.model.load_state_dict(torch.load(load_path))
        
    def forward(self, batch):
        g, _ = batch
        x = g.ndata['feat']
        mask = g.ndata['mask']
        y = g.ndata['label'][mask]
        #print(g)
        #print(x)
        _, logits = self.model(g, x)
        pred = logits[mask]
        loss = F.cross_entropy(pred, y)

        return loss, pred
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        
        # Logging to TensorBoard by default
        loss, _ = self.forward(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def trainGNN(train_dataloader, val_dataloader, in_feature_size, hidden_size, num_classes, epochs, batch_size):
    model = NodeGNN(in_feature_size, hidden_size, num_classes)

    save_path = os.path.join('node_gnn_save_double', '_'.join([str(in_feature_size), str(hidden_size), str('epochs')]))
    trainer = pl.Trainer(limit_train_batches=batch_size, max_epochs=epochs, accelerator="cuda", callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=5,verbose=False, mode='min')])
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
   
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path+'/model.pt')

    return model

def evaluate_model(model, dataset):
    count = 0

    for g, _ in tqdm(dataset, desc = 'testing'):
        with torch.no_grad():
            _, pred = model.model(g, g.ndata['feat'])

        mask = g.ndata['mask']
        # if pred.argmax(dim = -1) == g.ndata['label'][mask].argmax(dim = -1):
        #     count +=1
        # print(pred.argmax)
        # print(g.ndata['label'][mask])
        pred = pred[mask]
        x = pred.argmax(dim = -1)
        y = g.ndata['label'][mask]
        y = y.argmax(dim=-1)
        # print(x)
        # print(pred.argmax(dim = -1))
        # print(y)
        if x.item() == y.item():
            count += 1
    return count/len(dataset)


def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    label = g.ndata['label']
    mask = g.ndata['mask']
    #print(features.shape, label.shape, mask.shape)
    for e in range(5):
        # Forward
        logits = model(g, features)
        #print(logits.shape)
        # Compute prediction
        pred = logits.argmax(1)
        #print(pred.shape)

        #print(logits[mask].shape)
        #print(label[mask].shape)
        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[mask], label[mask])

        # # Compute accuracy on training/validation/test
        # train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        # val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        # test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # # Save the best validation accuracy and the corresponding test accuracy.
        # if best_val_acc < val_acc:
        #     best_val_acc = val_acc
        #     best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if e % 5 == 0:
        #     print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
        #         e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

if __name__ ==  '__main__':
    with open('verb_transformed_double_train.pkl','rb') as f:
        data = joblib.load(f)
    train_dataset = edsDataset(data)
    with open('verb_transformed_double_val.pkl','rb') as f:
        data = joblib.load(f)
    val_dataset = edsDataset(data)
    with open('verb_transformed_double_test.pkl','rb') as f:
        data = joblib.load(f)
    test_dataset = edsDataset(data)
    g,_ = train_dataset[0]

    train_sampler = SubsetRandomSampler(torch.arange(len(train_dataset)))
    train_dataloader = GraphDataLoader(train_dataset, sampler=train_sampler, batch_size=5, drop_last=False)

    val_dataloader = GraphDataLoader(val_dataset, batch_size=5, drop_last=False)

    model = trainGNN(train_dataloader, val_dataloader, train_dataset.num_features, 300, train_dataset.num_classes, 100, 32)

    print('test start')
    acc = evaluate_model(model, test_dataset)
    print(acc)
    # g = g.to('cuda')
    #model = GCN(g.ndata['feat'].shape[1], 300, dataset.num_classes).to('cuda')

    