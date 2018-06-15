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
import setting
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
from Vocab import Vocab_class   
from util import load_squad_data,create_mask,get_data_engine
from module import R_Net,decode
from metrics import batch_score

use_cuda = torch.cuda.is_available()
bug_flag=True

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='model.cpt', help='the model dir')
parser.add_argument('--batch_size', type=int, default=32, help='the size of batch [32]')
parser.add_argument('--char_input',type=int,default=1,help='use char encoder input?')
parser.add_argument('--emb_input',type=int,default=1,help='emb input? use DataLoader create embedding representation?')

parser.add_argument('--output_name',type=str,default='prediction_anser.json',help='the output name')
args = parser.parse_args()


if __name__ == '__main__':

    train_P,train_Q,train_P_c,train_Q_c,train_A,dev_P,dev_Q,dev_P_c,dev_Q_c,dev_A = load_squad_data()
 
    word_Vocab = Vocab_class()
    word_Vocab.load(setting.word_vocab_w2i_dir,setting.word_vocab_i2w_dir)
    word_embedding=np.load(setting.SQUAD_Word_Embedding_output_dir)

    #Get char Vocab and Embedding

    char_Vocab  = Vocab_class()
    if setting.use_all_char_vocab is True:
        char_Vocab.load(setting.char_all_vocab_w2i_dir   ,setting.char_all_vocab_i2w_dir)
        char_embedding=np.load(setting.SQUAD_Char_all_Embedding_output_dir)
    else:
        char_Vocab.load(setting.char_simple_vocab_w2i_dir,setting.char_simple_vocab_i2w_dir)
        char_embedding=np.load(setting.SQUAD_Char_simple_Embedding_output_dir)

    word_PAD_ID = word_Vocab.PAD_ID
    word_UNK_ID = word_Vocab.UNK_ID
    char_PAD_ID = char_Vocab.PAD_ID
    char_UNK_ID = char_Vocab.UNK_ID
    dev_P_mask   = create_mask(dev_P,word_PAD_ID,word_UNK_ID)
    dev_Q_mask   = create_mask(dev_Q,word_PAD_ID,word_UNK_ID)
    dev_P_char_mask   = create_mask(dev_P_c,char_PAD_ID,char_UNK_ID)
    dev_Q_char_mask   = create_mask(dev_Q_c,char_PAD_ID,char_UNK_ID)

    with open(setting.dev_Q_id_to_qid_dir,'rb') as f:
        dev_id_to_qid = pickle.load(f)
    dev_Q_id = np.load(setting.dev_Q_id_dir)

        
    print('Word Vocab size: %d | Char Vocab size: %d | Max context: %d | Max question: %d'%(
          word_embedding.shape[0],char_embedding.shape[0], dev_P.shape[1], dev_Q.shape[1]))
    valid_engine = get_data_engine(args.emb_input,  
                                            dev_P,  
                                            dev_Q,  
                                            dev_A,  
                                            dev_P_mask,  
                                            dev_Q_mask,  
                                            dev_P_c,  
                                            dev_Q_c,  
                                            dev_P_char_mask,  
                                            dev_Q_char_mask,  
                                            dev_Q_id,
                                            word_embedding,
                                            char_embedding,
                                            args.batch_size,
                                            use_cuda)



    R_net = torch.load(args.model_dir)

    if use_cuda:
        R_net = R_net.cuda()

    R_net.eval()
    valid_f1, valid_exact = 0, 0
    prediction_dict={}
    for p, q, ans_offset,p_mask,q_mask,pc,qc,pc_mask,qc_mask,Q_ids,idx in valid_engine:
        p = Variable(p).cuda() if use_cuda else Variable(p)
        q = Variable(q).cuda() if use_cuda else Variable(q)
        pc = Variable(pc).cuda() if use_cuda else Variabel(pc)
        qc = Variable(qc).cuda() if use_cuda else Variabel(qc)
        p_mask = Variable(p_mask).cuda() if use_cuda else Variable(p_mask)
        q_mask = Variable(q_mask).cuda() if use_cuda else Variable(q_mask)
        pc_mask = Variable(pc_mask).cuda() if use_cuda else Variable(pc_mask)
        qc_mask = Variable(qc_mask).cuda() if use_cuda else Variable(qc_mask)
        start_ans = Variable(ans_offset[:, 0]).cuda() if use_cuda else Variable(ans_offset[:, 0])
        end_ans = Variable(ans_offset[:, 1]).cuda() if use_cuda else Variable(ans_offset[:, 1])
        
        start,_,end,_ = R_net(p,q,p_mask,q_mask,pc,qc,pc_mask,qc_mask)
        
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
                    sentense +=  word_Vocab.get_word(dev_P[i,s_index-j])
                else:
                    sentense +=  word_Vocab.get_word(dev_P[i,s_index+j])
            prediction_dict[dev_id_to_qid[Q_ids[i]]] = sentense

    print('valid_f1: %f | valid_exact: %f'%(
          valid_f1/len(valid_engine), valid_exact/len(valid_engine)))



    with open(args.output_name,'w') as f:
        json.dump(prediction_dict,f)
        
