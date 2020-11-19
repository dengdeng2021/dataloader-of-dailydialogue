import time,datetime,os,codecs
#cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from transformers import BertModel,BertTokenizer

'''
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
    'num_workers':5,})# '''



def detect_whether_num_matching(data_df):
    # 检查是否有 标签数和文本数 不匹配 的情况
    n = []
    for i in range(data_df.shape[0]):
        len_cxt = len(data_df.iloc[i].SENT.strip().split('__eou__')[:-1])
        len_label = len(data_df.iloc[i].LABEL.strip().split(' '))
        if len_cxt!=len_label:
            n.append(i)
            print ('not matching in idx',i,len_cxt,len_label)
            print (data_dev_df.iloc[i].SENT)
            print (data_dev_df.iloc[i].LABEL)
    if len(n) == 0:
        print ('number matching: utterances and emotions')


def load_DailyDiag(path,n_class,source_type = 'train'):
    '''
    -------------------------------
    path = config.valid_file_src
    return:
        cxt_sents[i]:  dailydailogue 是变长
            [sent1,
             sent2,
             ...]
        cxt_sents_labels: 
            [[1,2,3,2,1,0]
             [1,1,1,2,2,2,1,1,1,0,0,5]
             ...]
    '''
    if source_type not in ['train','test','validation']:
        raise NameErrow('source_type',source_type)
    print ('\nloading data of ',source_type)
    with codecs.open(os.path.join(path,'dialogues_emotion_%s.txt'%source_type),'r') as doc:
        labels = doc.read().strip().split('\n')
    with codecs.open(os.path.join(path,'dialogues_%s.txt'%source_type),'r') as doc:
        lines = doc.read().strip().split('\n') # 句子之间分隔符： '__eou__'
        #lines = [dig.strip().split('__eou__') for dig in lines_]
    data_df = pd.DataFrame({'SENT':lines,'LABEL':labels})
    
    
    # 检查是否有对话轮数和标签个数不一致的情况
    detect_whether_num_matching(data_df)
    
    ### 统计会话长度
    labels = [[int(i) for i in l.strip().split(' ')] for l in labels] # 把 str list 转为 int
    len_diag = [len(l) for l in labels]
    print ('num of dialigues:',len(labels),len(lines))
    print ('max num of turns:',max(len_diag),
           '\nmin  num of turns:',min(len_diag),
           '\navg  num of turns:',sum(len_diag)/len(len_diag))
    
    ### 统计情感个数
    label_flatten = []
    for l in labels:
        label_flatten.extend(l)
    num_emos = []
    for i in range(n_class):
        num_emos.append(label_flatten.count(i))
    print ('total utterance:',sum(num_emos),',  total num of each emos',num_emos)

    return data_df

class DATA_HierarDailyDiag(Dataset):
    '''
    tokens = ['[CLS]', '一', '起', '吃', '饭', '啊', '[SEP]', '[PAD]', '[PAD]', '[PAD]']
    tokens_ids_tensor = tensor([ 101,  671, 6629, 1391, 7649, 1557,  102,    0,    0,    0])
    atten_mask = tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    label = tensor([0, 0, 1, 0, 0, 0, 0, 0]) 
    '''
    def __init__(self, data_df,config):
        #Store the contents of the file in a pandas dataframe
        self.df =  data_df# n_cxt 个句子
        self.pretrained_model_name = config.pretrained_model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
        
        self.max_cxt = config.n_cxt # 对话中保留的最长的 utternce个数
        self.maxlen = config.max_len_token # 句子中 max num of tokens
        self.categories = config.categories

    def __len__(self):
        return len(self.df) # 样本的个数
    
    def BERT_sent_preprocessing(self,sentence):
        '''
        tokenize, pad, mask,
        Preprocessing the text to be suitable for BERT
        return:
            tokens_ids, [101, 1962, 1737, 1557, 791, 1921, 102, 0, 0, 0]
            attn_mask,  [1 1 1 1 1 1 1 0 0 0]     
        '''
        tokens = self.tokenizer.tokenize(sentence) #Tokenize the sentence
        tokens = ['[CLS]'] + tokens + ['[SEP]'] #Insering the CLS and SEP token in the beginning and end of the sentence
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] #Padding sentences
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length
            
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids) #Converting the list to a pytorch tensor
        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long() # PAD 的 idx 为 0

        #return tokens_ids_tensor.unsqueeze(0),attn_mask.unsqueeze(0)
        return tokens_ids_tensor,attn_mask
    
    def pad_sent_propossing(self,n_pad):
        # 空句的 id 和 mask, pad n_pad 次
        tokens = ['[CLS]', '[SEP]'] + ['[PAD]'] * (self.maxlen-2)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids) #Converting the list to a pytorch tensor
        attn_mask = (tokens_ids_tensor != 0).long() # PAD 的 idx 为 0
        
        pad_tokens_ids_tensor = tokens_ids_tensor.expand(n_pad,self.maxlen)
        pad_attn_mask = attn_mask.expand(n_pad,self.maxlen)
        
        return pad_tokens_ids_tensor,pad_attn_mask
        
    def __getitem__(self, index):
        '''
        df.loc[index] 输入是多轮对话，pad 到 self.max_cxt 轮
      
        input:
            df.loc[index].SENT:  cxts = sent1__eou__sent2__eou__sent3__eou__
            df.loc[index].LABEL: label = '0 1 3'
        Return:
            tokens_ids 和 attn_mask 输入是 [PAD]
            label = [0,1,3,-1,-1,-1...]
            
            X1[0].shape = (10, 60)  # batch 中第一个样本n 句话的 word tokens 0-pad
            M1[0].shape = (10, 60)  # batch 中第一个样本 n 句话的 word tokens pad 是否 mask
            label[0].shape = (10, ) #  batch 中第一个样本 n 句话的标签 pad 部分为-1'''
        
        cxts = self.df.loc[index].SENT.strip().split('__eou__')[:-1]
        label = self.df.loc[index].LABEL
        label = torch.IntTensor([int(i) for i in  label.strip().split(' ')])
        
        if len(cxts)!=label.shape[0]:
            print ('DataMatchingError',index)
        # 截长 补短
        if len(cxts)>=self.max_cxt:
            whether_pad = False
            cxts,label = cxts[:self.max_cxt],label[:self.max_cxt]
        else:
            whether_pad = True
            n_pad = self.max_cxt - len(cxts)
            pad_tokens_ids_tensor,pad_attn_mask = self.pad_sent_propossing(n_pad)
            label = torch.cat((label, torch.IntTensor([-1]*n_pad)),0)
            
        ###################################################################
        tokens_ids = None
        for s in cxts:
            cur_tokens_ids,cur_attn_mask = self.BERT_sent_preprocessing(s)
            cur_tokens_ids,cur_attn_mask = cur_tokens_ids.unsqueeze(0),cur_attn_mask.unsqueeze(0)
            
            if (tokens_ids is None):
                tokens_ids,attn_mask = cur_tokens_ids,cur_attn_mask
            else:
                tokens_ids = torch.cat((tokens_ids,cur_tokens_ids), 0)
                attn_mask =  torch.cat((attn_mask,cur_attn_mask), 0)
        
        # pad
        if whether_pad:
                tokens_ids = torch.cat((tokens_ids,pad_tokens_ids_tensor), 0)
                attn_mask =  torch.cat((attn_mask,pad_attn_mask), 0)
     
        return tokens_ids,attn_mask,label