# dataloader-of-dailydialogue
在pytorch 中创建daily dialogue 的 dataloader
生成的数据主要有三项：

it, (X1,M1, labels) = enumerate(val_loader).__next__()
print (it, X1.shape,M1.shape,labels.shape)
  0 torch.Size([128, 10, 60]) torch.Size([128, 10, 60]) torch.Size([128, 10])

'''
X1[0].shape = (10, 60)  # batch 中第一个样本n 句话的 word tokens 0-pad
M1[0].shape = (10, 60)  # batch 中第一个样本 n 句话的 word tokens pad 是否 mask
label[0].shape = (10, ) #  batch 中第一个样本 n 句话的标签 pad 部分为-1
                        #[ 0,  0,  0,  0,  0,  0,  0, -1, -1, -1]

X1[0].shape = (10, 60) 
# X1[0] 是一轮对话中的10个句子，maxlen = 60，转换成 bert token idx （'bert-base-uncased'）
tensor([[  101,  2204,  2851,  1010,  2909,  1012,  2003,  2045,  1037,  2924,
          2379,  2182,  1029,   102,     0,     0,     0,     0,     0,     0,
            ...],
        [  101,  2045,  2003,  2028,  1012,  1019,  5991,  2185,  2013,  2182,
          1029,   102,     0,     0,     0,     0,     0,     0,     0,     0,
            ...],
        [  101,  2092,  1010,  2008,  1005,  1055,  2205,  2521,  1012,  2064,
          2017,  2689,  2070,  2769,  2005,  2033,  1029,   102,     0,     0,
            ...],
        [  101,  7543,  1010,  1997,  2607,  1012,  2054,  2785,  1997,  9598,
          2031,  2017,  2288,  1029,   102,     0,     0,     0,     0,     0,
            ...],
        [  101, 19395,  1012,   102,     0,     0,     0,     0,     0,     0,
           ...],
        [  101,  2129,  2172,  2052,  2017,  2066,  2000,  2689,  1029,   102,
             ...],
        [  101,  6694, 11237,  1012,  2182,  2017,  2024,  1012,   102,     0,
            ...],
        [  101,   102,     0,     0,     0,     0,     0,     0,     0,     0,
            ...],
        [  101,   102,     0,     0,     0,     0,     0,     0,     0,     0,
             ...],
        [  101,   102,     0,     0,     0,     0,     0,     0,     0,     0,
           ...]])
    
M1[0].shape = (10, 60)  
# M1[0] 是一轮对话中的10个句子，maxlen = 60，0 表示该处token 为 0-pad
    tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
label[0] = [ 0,  1,  0,  3,  3,  1,  4, -1, -1, -1]'''
# label[0] 是一轮对话中的10个句子对应的情感idx，-1为pad
dailydialogue 中共 6个情感+1个Neutral，tag idx 范围为 0-6
