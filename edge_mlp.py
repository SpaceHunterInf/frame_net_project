import joblib
import torch
from torch import optim, nn, utils, Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
import os, sys
from dgl_data import edgeDataset
class MLP(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers= 5):
        super().__init__()
        self.input_layer = torch.nn.Linear(in_feats, h_feats)
        self.hidden_layers = [torch.nn.Linear(h_feats, h_feats) for x in range(num_layers)]
        self.output_layer = torch.nn.Linear(h_feats, num_classes)
        self.layers = [self.input_layer] + self.hidden_layers + [self.output_layer]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, in_feats):
        y = self.layers(in_feats)
        return y

class EdgeMLP(pl.LightningModule):
    def __init__(self, in_feats, h_feats, num_classes, load=False, load_path=None):
        super().__init__()
        self.model = MLP(in_feats, h_feats, num_classes)
        if load:
            self.model.load_state_dict(torch.load(load_path))
        
    def forward(self, batch):
        input_dict = batch
        x = input_dict['feat']
        y = input_dict['label']
        #print(g)
        #print(x)
        pred = self.model(x)
        # print(pred)
        # print(y)
        # print(x.shape)
        # print(y.shape)
        # print(pred.shape)
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


def trainMLP(train_dataloader, val_dataloader, in_feature_size, hidden_size, num_classes, epochs, batch_size):
    model = EdgeMLP(in_feature_size, hidden_size, num_classes)

    save_path = os.path.join('edge_mlp_double_save', '_'.join([str(in_feature_size), str(hidden_size), str('epochs')]))
    trainer = pl.Trainer(limit_train_batches=batch_size, max_epochs=epochs, accelerator="cuda", callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=5,verbose=False, mode='min')])
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
   
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path+'/model.pt')

    return model

def evaluate_model(model, dataset):
    count = 0

    for d in tqdm(dataset, desc = 'testing'):
        with torch.no_grad():
            pred = model.model(d['feat'].to('cuda'))

        x = pred.argmax(dim = -1)
        y = d['label']
        y = y.argmax(dim=-1)
        # print(x)
        # print(pred.argmax(dim = -1))
        # print(y)
        if x.item() == y.item():
            count += 1

        print(x.item(), y.item())
    return count/len(dataset)


if __name__ ==  '__main__':
    with open('edge_transformed_double_train.pkl','rb') as f:
        data = joblib.load(f)
    train_dataset = edgeDataset(data)
    with open('edge_transformed_double_dev.pkl','rb') as f:
        data = joblib.load(f)
    val_dataset = edgeDataset(data)
    with open('edge_transformed_double_test.pkl','rb') as f:
        data = joblib.load(f)
    test_dataset = edgeDataset(data)
    
    print(train_dataset[0]['feat'].shape)
    print(train_dataset[0]['label'].shape)
    print('---')
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = trainMLP(train_dataloader, val_dataloader, train_dataset.num_features, 300, train_dataset.num_classes, 100, 32)

    print('test start')
    model.to('cuda')
    acc = evaluate_model(model, test_dataset)
    print(acc)
    # g = g.to('cuda')
    #model = GCN(g.ndata['feat'].shape[1], 300, dataset.num_classes).to('cuda')