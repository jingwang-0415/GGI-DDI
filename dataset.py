import os
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader.data_list_loader import DataListLoader
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import numpy as np
from tqdm import tqdm


def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
def save_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {path}!')






def split_train_valid(data_df, fold, val_ratio=0.2):
        cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
        train_index, val_index = next(iter(cv_split.split(X=range(len(data_df)), y = data_df['Y'])))

        train_df = data_df.iloc[train_index]
        val_df = data_df.iloc[val_index]

        return train_df, val_df

def mycola(datalist):
    head_list = [data[0] for data in datalist]
    head_list = sum(head_list,[])
    tail_list = [data[1] for data in datalist]
    tail_list = sum(tail_list,[])
    rel_list = [data[2] for data in datalist]
    rel_list = sum(rel_list,[])
    label_list = [data[3] for data in datalist]
    label_list = sum(label_list,[])
    return head_list,tail_list,rel_list,label_list
class mydatalist(DataLoader):
    def __init__(self,dataset,batch_size,shuffle,**kwargs):

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,collate_fn=dataset.collate_fn,
                         **kwargs)
class DDI_dataset(Dataset):
    def __init__(self,drug_graph,df_path):
        self.pair_path = df_path
        self.data_df = pd.read_csv(df_path)

        self.drug_graph = drug_graph
    def __len__(self):
        return len(self.data_df)
    def __getitem__(self, item):

        row = self.data_df.iloc[item]

        return row


    def collate_fn(self,batch):
        head_list = []
        tail_list = []
        label_list = []
        rel_list = []
        for row in batch:

            Drug1_ID, Drug2_ID, Y, Neg_samples = row['Drug1_ID'], row['Drug2_ID'], row['Y'], row['Neg samples']
            Neg_ID, Ntype = Neg_samples.split('$')
            # Neg_ID =Neg_samples
            h_graph = self.drug_graph.get(Drug1_ID)
            t_graph = self.drug_graph.get(Drug2_ID)
            n_graph = self.drug_graph.get(Neg_ID)

            pos_pair_h = h_graph
            pos_pair_t = t_graph
            # neg_pair_h = n_graph
            # neg_pair_t = t_graph
            if Ntype == 'h':
                neg_pair_h = n_graph
                neg_pair_t = t_graph
            else:
                neg_pair_h = h_graph
                neg_pair_t = n_graph

            head_list.append(pos_pair_h)
            head_list.append(neg_pair_h)
            tail_list.append(pos_pair_t)
            tail_list.append(neg_pair_t)

            rel_list.append(torch.LongTensor([Y]))
            rel_list.append(torch.LongTensor([Y]))

            label_list.append(torch.FloatTensor([1]))
            label_list.append(torch.FloatTensor([0]))
        head_pairs = Batch.from_data_list(head_list, follow_batch=['edge_index'])
        tail_pairs = Batch.from_data_list(tail_list, follow_batch=['edge_index'])
        rel = torch.cat(rel_list, dim=0)
        label = torch.cat(label_list, dim=0)

        return head_pairs, tail_pairs, rel, label




class twosides_pkl(Dataset):
    def __init__(self, data_dir,):
        self.DDI_index = []
        fileIn = open(data_dir)
        line = fileIn.readline()
        while line:
            id1, id2, smiles1, smiles2, interaction, label = line.strip().split()
            line = fileIn.readline()
            self.DDI_index.append([id1, id2, interaction, label])
        with open(f'./drug_data.pkl', 'rb') as f:
            self.drugdata = pickle.load(f)
        # with open(data_dir, 'rb') as f:
        #     data = pickle.load(f)
        # self.data = data
    def __len__(self):
        return len(self.DDI_index)
    def __getitem__(self, item):
        row = self.DDI_index[item]
        return row

    def collate_fn(self, batch):
        head_list = []
        tail_list = []
        label_list = []
        rel_list = []
        for data in batch:
            drug1, drug2, interaction, label = int(data[0]),int(data[1]),int(data[2]),int(data[3])
            # a = data['drug_1']
            # 
            # head_list.append(data['drug_1'])
            # 
            # tail_list.append(data['drug_2'])
            x_1, edge_index_1, edge_feature_1 = self.drugdata[drug1]

            Drug1 = Data(x=x_1, edge_index=edge_index_1, edge_attr=edge_feature_1)
            x_2, edge_index_2, edge_feature_2 = self.drugdata[drug2]

            Drug2 = Data(x=x_2, edge_index=edge_index_2, edge_attr=edge_feature_2)
            head_list.append(Drug1)
            tail_list.append(Drug2)
            label_list.append(torch.FloatTensor([label]))
            rel_list.append(torch.LongTensor([interaction]))
        head_pairs = Batch.from_data_list(head_list, follow_batch=['edge_index'])
        tail_pairs = Batch.from_data_list(tail_list, follow_batch=['edge_index'])
        rel = torch.cat(rel_list, dim=0)
        label = torch.cat(label_list, dim=0)

        return head_pairs, tail_pairs, rel, label

class twosides_pkl_loader(DataLoader):
    def __init__(self,dataset,batch_size,shuffle,**kwargs):

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,collate_fn=dataset.collate_fn,
                         **kwargs)

def prepare_twosides(dir):
    return twosides_pkl(dir)



class load_pkl(Dataset):
    def __init__(self,dataset,type,batchsize):
        self.dataset = dataset
        self.type = type
        self.batchsize = batchsize
        self.data_dir = os.path.join(dataset,type,batchsize)
        self.data_files = [
            f for f in os.listdir(self.data_dir) if f.endswith('.pkl')
        ]
        self.data_files.sort()
    def __len__(self):
        return len(self.data_files)
    def __getitem__(self, index):

        with open(os.path.join(self.data_dir, self.data_files[index]),
                  'rb') as f:
            reaction_data = pickle.load(f)
        head_pairs = reaction_data['head_pairs']
        tail_pairs = reaction_data['tail_pairs']
        rel = reaction_data['rel']
        label = reaction_data['label']
        return head_pairs, tail_pairs, rel, label
    def collate_fn(self,data):
        head_pairs, tail_pairs, rel, label = map(list, zip(*data))
        return head_pairs[0], tail_pairs[0], rel[0], label[0]


