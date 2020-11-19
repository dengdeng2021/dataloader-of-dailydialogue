# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import os
import datetime
import unicodedata

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

config = AttrDict()
config.update({
    # data:
    'train_file_src':'/home/djw/Experiments/all_corpus/16_ijcnlp_dailydialog/train',
    'test_file_src':'/home/djw/Experiments/all_corpus/16_ijcnlp_dailydialog/test',
    'valid_file_src':'/home/djw/Experiments/all_corpus/16_ijcnlp_dailydialog/validation',
    'max_len_token':60,
    'n_cxt':10,
    'emo_tags': ['Neutral','anger','disgust','fear','happiness','sadness','surprise'],
    'categories': 7,    
    # pretrained model name:
    'pretrained_model_name':'bert-base-uncased', # uncased 全部转成小写，cased：区分大小写
    'freeze_bert':True,
    'bert_hsz': 768,
    'batch_size':128,
    'num_workers':5,})


from utils_dailydialogue_dataloader import load_DailyDiag,DATA_HierarDailyDiag
########################################################
data_train_df = load_DailyDiag(config.train_file_src,config.categories,
                               source_type = 'train')
data_dev_df = load_DailyDiag(config.valid_file_src,config.categories,
                             source_type = 'validation')
data_test_df = load_DailyDiag(config.test_file_src,config.categories,
                              source_type = 'test')
print (data_train_df.shape,data_dev_df.shape,data_test_df.shape)
print ('\n',data_dev_df[:5])

########################################################
############################
train_set = DATA_HierarDailyDiag(data_df = data_train_df,config = config)
train_loader = DataLoader(train_set, 
                          batch_size = config.batch_size,
                          num_workers = config.num_workers)
############################
val_set = DATA_HierarDailyDiag(data_df = data_dev_df,config = config)
val_loader = DataLoader(val_set,
                        batch_size = config.batch_size, 
                        num_workers = config.num_workers)
############################
test_set = DATA_HierarDailyDiag(data_df = data_test_df,config = config)
test_loader = DataLoader(test_set,
                        batch_size = config.batch_size, 
                        num_workers = config.num_workers)

########################################################
# for it, (X1,M1, labels) in enumerate(val_loader):
#     print (X1.shape,M1.shape,labels.shape)
it, (X1,M1, labels) = enumerate(val_loader).__next__()
print (it, X1.shape,M1.shape,labels.shape)
# print (X1[1,:5],M1[1,:5])
print (labels[:5])

'''
loading data of  train
number matching: utterances and emotions
num of dialigues: 11118 11118
max num of turns: 35 
min  num of turns: 2 
avg  num of turns: 7.8404389278647235
total utterance: 87170 ,  total num of each emos [72143, 827, 303, 146, 11182, 969, 1600]

loading data of  validation
number matching: utterances and emotions
num of dialigues: 1000 1000
max num of turns: 31 
min  num of turns: 2 
avg  num of turns: 8.069
total utterance: 8069 ,  total num of each emos [7108, 77, 3, 11, 684, 79, 107]

loading data of  test
number matching: utterances and emotions
num of dialigues: 1000 1000
max num of turns: 26 
min  num of turns: 2 
avg  num of turns: 7.74
total utterance: 7740 ,  total num of each emos [6321, 118, 47, 17, 1019, 102, 116]
(11118, 2) (1000, 2) (1000, 2)

                                                 SENT  \
0  Good morning , sir . Is there a bank near here...   
1  Good afternoon . This is Michelle Li speaking ...   
2  What qualifications should a reporter have ? _...   
3  Hi , good morning , Miss ? what can I help you...   
4  Excuse me , ma'am . Can you tell me where the ...   

                              LABEL  
0                    0 0 0 0 0 0 0   
1              0 0 0 0 0 0 0 0 0 0   
2                          0 0 0 0   
3  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4   
4                  0 0 0 0 0 0 4 4   
0 torch.Size([128, 10, 60]) torch.Size([128, 10, 60]) torch.Size([128, 10])
tensor([[ 0,  0,  0,  0,  0,  0,  0, -1, -1, -1],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0, -1, -1, -1, -1, -1, -1],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  4,  4, -1, -1]], dtype=torch.int32)
'''