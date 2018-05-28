import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
import data
import random
import datetime
import pickle
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
from Vocab import Vocab_SQUAD
from module import R_Net,decode
from metrics import batch_score
use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='model.cpt', help='the model dir')
parser.add_argument('--output_name',type=str,default='prediction_anser.json',help='the output name')
args = parser.parse_args()


if __name__ == '__main__':
    dev_P_dir = r"./SQUAD/dev/dev_P.npy"
    dev_Q_dir = r"./SQUAD/dev/dev_Q.npy"
    dev_A_dir = r"./SQUAD/dev/dev_A.npy"
    dev_Q_id_dir = r"./SQUAD/dev/dev_Q_id.npy"
    dev_Q_id_to_qid_dir=r"./SQUAD/dev/dev_id_to_qid.pkl"

    Embedding_output_dir=r"./SQUAD/squad_word_embedding.npy"

    dev_P = np.load(dev_P_dir)
    dev_Q = np.load(dev_Q_dir)
    dev_A = np.load(dev_A_dir)
    dev_Q_id = np.load(dev_Q_id_dir)

    dev_P_mask=np.zeros(dev_P.shape,dtype=np.uint8)
    dev_Q_mask=np.zeros(dev_Q.shape,dtype=np.uint8)


    SQUAD_Vocab = Vocab_SQUAD()
    SQUAD_Vocab.load()
    PAD_ID = SQUAD_Vocab.PAD_ID
    UNK_ID = SQUAD_Vocab.UNK_ID

    dev_P_mask[dev_P==PAD_ID] = 1
    dev_P_mask[dev_P==UNK_ID] = 0
    dev_Q_mask[dev_Q==PAD_ID] = 1
    dev_Q_mask[dev_Q==UNK_ID] = 0   


    with open(dev_Q_id_to_qid_dir,'rb') as f:
        dev_id_to_qid = pickle.load(f)
    embedding=np.load(Embedding_output_dir)
        

    print('Embedding size:',embedding.shape)

    print('Vocab size: %d | Max context: %d | Max question: %d'%(
          embedding.shape[0], dev_P.shape[1], dev_Q.shape[1]))
    valid_engine = DataLoader(data.DataEngine_for_prediction(dev_P,
                                              dev_Q,
                                              dev_A,
                                              dev_P_mask,
                                              dev_Q_mask,
                                              embedding,
                                              dev_Q_id),
                              batch_size=10,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=use_cuda)

    R_net = torch.load(args.model_dir)

    if use_cuda:
        R_net = R_net.cuda()

    R_net.eval()
    valid_f1, valid_exact = 0, 0
    prediction_dict={}
    for p, q, ans_offset,p_mask,q_mask,Q_ids,idx in valid_engine:
        p = Variable(p).cuda() if use_cuda else Variable(p)
        q = Variable(q).cuda() if use_cuda else Variable(q)
        p_mask = Variable(p_mask).cuda() if use_cuda else Variable(p_mask)
        q_mask = Variable(q_mask).cuda() if use_cuda else Variable(q_mask) 
        start_ans = Variable(ans_offset[:, 0]).cuda() if use_cuda else Variable(ans_offset[:, 0])
        end_ans = Variable(ans_offset[:,1]).cuda() if use_cuda else Variable(ans_offset[:, 1])
        start,_,end,_ = R_net(p,q,p_mask,q_mask)
        
        start, end, scores = decode(start.data.cpu(), end.data.cpu(), 1)
        f1_score, exact_match_score = batch_score(start, end, ans_offset)
        valid_f1 += f1_score
        valid_exact += exact_match_score

        for i in range(p.size(0)):
            s_index = int(start[i])
            e_index = int(end[i])
            ans_len = abs(e_index-s_index)+1
            sentense= ""
            for j in range(ans_len):
                if j!=0:
                    sentense+=" "
                if s_index>=e_index:
                    sentense +=  SQUAD_Vocab.get_word(dev_P[i,s_index-j])
                else:
                    sentense +=  SQUAD_Vocab.get_word(dev_P[i,s_index+j])
            prediction_dict[dev_id_to_qid[Q_ids[i]]] = sentense

    print('valid_f1: %f | valid_exact: %f'%(
          valid_f1/len(valid_engine), valid_exact/len(valid_engine)))



    with open(args.output_name,'w') as f:
        json.dump(prediction_dict,f)
        
