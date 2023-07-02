import os.path
import pickle
import torch
from dataset import read_pickle,mydatalist,prepare_twosides,twosides_pkl_loader,load_pkl
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, precision_score, f1_score,recall_score
from sklearn.metrics import precision_recall_curve,auc,accuracy_score,average_precision_score
import numpy as np
from tqdm import tqdm
from util import log_util

from torch import optim
from DDI import DD_Pre
import time
import faulthandler
faulthandler.enable()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_ids = range(torch.cuda.device_count())
torch.multiprocessing.set_sharing_strategy('file_system')




def test_DDI(test_loader, model):
    model.eval()
    y_pred = []
    y_label = []
    with torch.no_grad():
        bar = tqdm(test_loader, ncols=80)
        for i,batches in enumerate(bar):

            head_list, tail_list, rel_list, Label = [data.to(device) for data in batches]
            predictions= model(head_list, tail_list, rel_list,False)
            predictions = predictions.squeeze()

            predictions = torch.sigmoid(predictions)

            predictions = predictions.detach().cpu().numpy()
            Label = Label.detach().cpu().numpy()
            y_label = y_label + Label.flatten().tolist()
            y_pred = y_pred + predictions.flatten().tolist()
    y_pred1 = np.array(y_pred)
    y_label1 = np.array(y_label)
    y_pred1_label = (y_pred1>=0.5).astype(np.int32)
    roc_test_ACC,roc_test_AUROC,f1,roc_test_Pre,recall,roc_test_AUPR = accuracy_score(y_label1,y_pred1_label),roc_auc_score(y_label, y_pred1),f1_score(y_label1,y_pred1_label), precision_score(y_label1, y_pred1_label),recall_score(y_label1, y_pred1_label),average_precision_score(y_label1,y_pred1,average='micro')
    p, r, t = precision_recall_curve(y_label1, y_pred1)
    roc_test_AUC = auc(r, p)
    return  roc_test_ACC,roc_test_AUC,f1,roc_test_Pre,recall,roc_test_AUPR,roc_test_AUROC
if __name__ == '__main__':
    dataset = 'twosides'
    logs = log_util(dataset,'0.2')
    for fold in [1]:

        cold_train_dir = './cold_train.txt'
        cold_c2_dir = './cold_test_C2.txt'
        cold_c3_dir = './cold_test_C3.txt'
        cold_train_data = prepare_twosides(cold_train_dir)
        cold_c2_data = prepare_twosides(cold_c2_dir)
        cold_c3_data = prepare_twosides(cold_c3_dir)

        if torch.cuda.is_available():

            model = DD_Pre(45,0.2,0.2).cuda()


        train_loader = twosides_pkl_loader(cold_train_data,batch_size=2048,shuffle=True,num_workers=2,pin_memory = True)

        c2 = twosides_pkl_loader(cold_c2_data,batch_size = 1024,shuffle=False,num_workers=2)

        c3 = twosides_pkl_loader(cold_c3_data,batch_size = 1024,shuffle=False,num_workers=2)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))


        loss_history = []

        t_total=time.time()

        epochs=81

        for epoch in range(0,epochs):

            model.train()
            t = time.time()
            y_pred_train = []
            y_label_train = []
            bar =  tqdm(train_loader,ncols=80)
            total_loss = 0
            batch = 0
            for i, batches in enumerate(bar):
                bar.set_description('Epoch ' + str(epoch))

                head_list, tail_list, rel_list,Label = [data.to(device) for data in batches]
                predictions, dis_loss = model(head_list, tail_list, rel_list,True)

                predictions = predictions.squeeze()
                loss1 = torch.nn.BCEWithLogitsLoss(reduction='sum')(predictions, Label)
                loss = loss1
                optimizer.zero_grad()
                loss1.backward()
                optimizer.step()

                predictions = torch.sigmoid(predictions)
                predictions = predictions.detach().cpu().numpy()
                Label = Label.detach().cpu().numpy()
                y_label_train = y_label_train + Label.flatten().tolist()
                y_pred_train = y_pred_train + predictions.flatten().tolist()
                total_loss += loss.item()
                batch = len(y_label_train)
                bar.set_postfix(loss ='%.5f' %(total_loss/batch))
            y_pred_train = np.array(y_pred_train)
            y_pred_train_label = (y_pred_train>=0.5).astype(np.int32)
            y_label_train = np.array(y_label_train)
            roc_train_ACC, roc_train_AUROC, train_f1, roc_train_Pre, train_recall, roc_train_AUPR = accuracy_score(y_label_train,y_pred_train_label),roc_auc_score(y_label_train, y_pred_train),f1_score(y_label_train,y_pred_train_label), precision_score(y_label_train, y_pred_train_label),recall_score(y_label_train, y_pred_train_label),average_precision_score(y_label_train,y_pred_train,average='micro')
            p,r,t = precision_recall_curve(y_label_train,y_pred_train)
            roc_train_AUC = auc(r,p)

            print(roc_train_AUC)
            logs.save_log(epoch,roc_train_ACC, roc_train_AUC, train_f1, roc_train_Pre, train_recall, roc_train_AUPR,roc_train_AUROC,'train',model,optimizer)

            if epoch % 2 == 0:

                roc_test_ACC, roc_test_AUC, f1, roc_test_Pre, recall, roc_test_AUPR,roc_test_AUROC = test_DDI(c2, model)
                logs.save_log(epoch,roc_test_ACC,roc_test_AUC,f1,roc_test_Pre,recall,roc_test_AUPR,roc_test_AUROC,'c2',model,optimizer)
                print(roc_test_Pre)
                roc_test_ACC, roc_test_AUC, f1, roc_test_Pre, recall, roc_test_AUPR,roc_test_AUROC = test_DDI(c3, model)
                logs.save_log(epoch,roc_test_ACC,roc_test_AUC,f1,roc_test_Pre,recall,roc_test_AUPR,roc_test_AUROC,'c3',model,optimizer)
                print(roc_test_Pre)


        torch.save(model.state_dict(),
               './save/twosides/{}_checkpoint.pt'.format('last'))





