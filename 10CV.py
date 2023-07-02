import pickle
import os
import random
from random import randint
from sklearn.model_selection import KFold
from random import shuffle
from tqdm import tqdm
from torch_geometric.data import  Data

#1、导入数据
def load_Smiles():
    Datas = './CID_smiles.txt'
    Smiles = []
    fileIn = open(Datas)
    line = fileIn.readline()
    while line:
        lineArr = line.strip().split()
        Smiles.append(lineArr[1])
        line = fileIn.readline()
    return Smiles

def neg():
    neg = []
    neg1 = randint(0,554)
    neg2 = randint(0,554)
    inte = randint(0,1317)
    lab = 0
    neg.append(neg1)
    neg.append(neg2)
    neg.append(inte)
    neg.append(lab)
    return neg

def load_DDIs():
    Datas = './DDI.txt'
    DDIs = []
    fileIn = open(Datas)
    line = fileIn.readline()
    while line:
        lineArr = line.strip().split()
        lineArr.append(1)
        DDIs.append(list(map(int,lineArr)))
        DDIs.append(neg())
        line = fileIn.readline()
    return DDIs

data_loader = 'train'
def write_to_txt():
    DDs = load_DDIs()
    shuffle(DDs)
    Smiles = load_Smiles()
    kf = KFold(10, shuffle=True, random_state=1)
    flag = 0
    for i, (trian_id, test_id) in tqdm(enumerate(kf.split(DDs))):
        print(trian_id)
        for ids in trian_id:
            DD = DDs[ids]
            i_str = str(i+1)
            flag += 1
            filename = 'train'+i_str+'.txt'
            with open(filename, 'a') as f1:
                f1.write(str(DD[0]) + ' ' + str(DD[1]) + ' ' + Smiles[DD[0]] + ' ' + Smiles[DD[1]] + ' ' + str(DD[2]) + ' ' + str(DD[3]) + '\n')

        for ids in test_id:
            DD = DDs[ids]
            i_str = str(i + 1)
            filename = 'test' + i_str + '.txt'
            with open(filename, 'a') as f2:
                f2.write(str(DD[0]) + ' ' + str(DD[1]) + ' ' + Smiles[DD[0]] + ' ' + Smiles[DD[1]] + ' ' + str(DD[2]) + ' ' + str(DD[3]) + '\n')

    return 0


#3、导入以上DDI数据，转换成可训练的pkl文件
def load_train_DDI(i):
    Datas = './txt/train{}.txt'.format(i)
    DDI_index = []
    fileIn = open(Datas)
    line = fileIn.readline()
    while line:
        id1, id2, smiles1, smiles2, interaction,label = line.strip().split()
        line = fileIn.readline()
        DDI_index.append(map(int,[id1, id2, interaction, label]))
    return DDI_index
def load_test_DDI(i):
    Datas = './txt/test{}.txt'.format(i)
    DDI_index = []
    fileIn = open(Datas)
    line = fileIn.readline()
    while line:
        id1, id2, smiles1, smiles2, interaction,label = line.strip().split()
        line = fileIn.readline()
        DDI_index.append(map(int,[id1, id2, interaction, label]))

    return DDI_index
def save_data(data, filename):
    dirname = f'./'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = dirname + '/' + filename
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')

def load_DD_data():
    with open(f'./drug_data.pkl', 'rb') as f:
        drugdata = pickle.load(f)

    for i in range(1,11):
        Drug_pair = {}
        DDI_index = load_train_DDI(i)
        for index, (drug1, drug2, interaction, label) in enumerate(DDI_index):
            x_1, edge_index_1, edge_feature_1 = drugdata[drug1]
            Drug1 = Data(x=x_1, edge_index=edge_index_1, edge_attr=edge_feature_1)
            x_2, edge_index_2, edge_feature_2 = drugdata[drug2]
            Drug2 = Data(x=x_2, edge_index=edge_index_2, edge_attr=edge_feature_2)
            drugpair = dict(drug_1=Drug1, drug_2=Drug2, Inter=interaction, Label=label)
            Drug_pair[index] = drugpair

        save_data(Drug_pair, 'train{}.pkl'.format(i))
        Drug_pair = {}
        DDI_index = load_test_DDI(i)
        for index, (drug1, drug2, interaction, label) in enumerate(DDI_index):
            x_1, edge_index_1, edge_feature_1 = drugdata[drug1]
            Drug1 = Data(x=x_1, edge_index=edge_index_1, edge_attr=edge_feature_1)
            x_2, edge_index_2, edge_feature_2 = drugdata[drug2]
            Drug2 = Data(x=x_2, edge_index=edge_index_2, edge_attr=edge_feature_2)
            drugpair = dict(drug_1=Drug1, drug_2=Drug2, Inter=interaction, Label=label)
            Drug_pair[index] = drugpair

        save_data(Drug_pair, 'test{}.pkl'.format(i))

    return Drug_pair

# a = load_DD_data()
a = write_to_txt()






