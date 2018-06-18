import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
import data
import random
import datetime
import pickle
import numpy as np
import json
import glob
import torch
import torch.nn as nn
import setting
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
from Vocab import Vocab_class   
from util import load_squad_data,create_mask,get_data_engine,resolve_file_name
from module import R_Net,decode
from metrics import batch_score

use_cuda = torch.cuda.is_available()
bug_flag=True

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='model.cpt', help='the model dir')
parser.add_argument('--batch_size', type=int, default=32, help='the size of batch [32]')
parser.add_argument('--data_version',type=int,default=1,help='choose squad database vesrion 1 or 2 or DRCD is 3')
parser.add_argument('--output_name',type=str,default='prediction_answer',help='the output name')
parser.add_argument('--model_list_dir',type=str,default='None',help='list of Model.cpt dir!')

args = parser.parse_args()


if __name__ == '__main__':
    if args.data_version==1:
        Model_base = setting.Model_v1_dir
        SQUAD_dir  = setting.SQUAD_v1_dir
        train_dir  = setting.Train_v1_dir
        dev_dir    = setting.DEV_v1_dir
        Predict_base= setting.Prediction_v1_dir
        print('choose SQUAD v1.1 Dataset')
    elif args.data_version==2:
        Model_base = setting.Model_v2_dir
        SQUAD_dir  = setting.SQUAD_v2_dir
        train_dir  = setting.Train_v2_dir
        dev_dir    = setting.DEV_v2_dir
        Predict_base= setting.Prediction_v2_dir
        print('choose SQUAD v2.0 Dataset')
    elif args.data_version==3:
        Model_base = setting.DRCD_model_dir
        SQUAD_dir  = setting.DRCD_dir
        train_dir  = setting.DRCD_train_dir
        dev_dir    = setting.DRCD_dev_dir
        Predict_base= setting.DRCD_Prediction_dir
        print('choose DRCD Dataset')
        setting.use_all_char_vocab=True
    else:
        Model_base = setting.Model_v1_dir
        SQUAD_dir  = setting.SQUAD_v1_dir
        train_dir  = setting.Train_v1_dir
        dev_dir    = setting.DEV_v1_dir
        Predict_base= setting.Prediction_v1_dir
        print('not this version,Auto choose SQUAD v1.1 dataset.')
    train_P,train_Q,train_P_c,train_Q_c,train_A,dev_P,dev_Q,dev_P_c,dev_Q_c,dev_A = load_squad_data(version_flag=args.data_version)
 
    word_Vocab = Vocab_class()
    word_Vocab.load(os.path.join(SQUAD_dir,setting.word_vocab_w2i_file),os.path.join(SQUAD_dir,setting.word_vocab_i2w_file))
    word_embedding=np.load(os.path.join(SQUAD_dir,setting.SQUAD_Word_Embedding_file))

    #Get char Vocab and Embedding

    char_Vocab  = Vocab_class()
    if setting.use_all_char_vocab is True:
        char_Vocab.load(os.path.join(SQUAD_dir,setting.char_all_vocab_w2i_file) ,os.path.join(SQUAD_dir,setting.char_all_vocab_i2w_file))
        char_embedding=np.load(os.path.join(SQUAD_dir,setting.SQUAD_Char_all_Embedding_file))
    else:
        char_Vocab.load(os.path.join(SQUAD_dir,setting.char_simple_vocab_w2i_file),os.path.join(SQUAD_dir,setting.char_simple_vocab_i2w_file))
        char_embedding=np.load(os.path.join(SQUAD_dir,setting.SQUAD_Char_simple_Embedding_file))


    word_PAD_ID = word_Vocab.PAD_ID
    word_UNK_ID = word_Vocab.UNK_ID
    char_PAD_ID = char_Vocab.PAD_ID
    char_UNK_ID = char_Vocab.UNK_ID
    dev_P_mask   = create_mask(dev_P,word_PAD_ID,word_UNK_ID)
    dev_Q_mask   = create_mask(dev_Q,word_PAD_ID,word_UNK_ID)
    dev_P_char_mask   = create_mask(dev_P_c,char_PAD_ID,char_UNK_ID)
    dev_Q_char_mask   = create_mask(dev_Q_c,char_PAD_ID,char_UNK_ID)
    
    with open(os.path.join(dev_dir,setting.dev_Q_id_to_qid_file),'rb') as f:
        dev_id_to_qid = pickle.load(f)
    
    dev_Q_id = np.load(os.path.join(dev_dir,setting.dev_Q_id_file))

        
    print('Word Vocab size: %d | Char Vocab size: %d | Max context: %d | Max question: %d'%(
          word_embedding.shape[0],char_embedding.shape[0], dev_P.shape[1], dev_Q.shape[1]))
    valid_engine_1 = get_data_engine(1,  
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
    valid_engine_0 = get_data_engine(0,  
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

    if args.model_list_dir!='None':
        real_list=glob.glob(args.model_list_dir)
        for model_dir in real_list:
            print('now_propcessing_for_this_module:'+str(model_dir))
            mode,epoch,_,_,_,char_input,emb_input,concat,hidden,batch_size  = resolve_file_name(model_dir)
            R_net = torch.load(model_dir)
            if use_cuda:
                R_net = R_net.cuda()
            R_net.eval()
            valid_f1, valid_exact = 0, 0
            prediction_dict={}
            if emb_input==1:
                valid_engine=valid_engine_1
            else:
                valid_engine=valid_engine_0
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
                        if j!=0 and args.data_version!=3:
                            sentense+=" "
                        if s_index>=e_index:
                            sentense +=  word_Vocab.get_word(dev_P[i,s_index-j])
                        else:
                            sentense +=  word_Vocab.get_word(dev_P[i,s_index+j])
                    prediction_dict[dev_id_to_qid[dev_Q_id[i]]] = sentense

            print('valid_f1: %f | valid_exact: %f'%(
                  valid_f1/len(valid_engine), valid_exact/len(valid_engine)))

            with open(Predict_base+r'/'+args.output_name+'_for_char_input'+str(char_input)+'_emb_input_'+str(emb_input)+'_concat_'+str(concat)+'_hidden_'+str(hidden)+'_batch_size_'+str(batch_size)+'_'+str(mode)+'_epoch_'+str(epoch)+'.json','w') as f:
                json.dump(prediction_dict,f)
    else:
        mode,epoch,_,_,_,char_input,emb_input,concat,hidden,batch_size  = resolve_file_name(model_dir)
        R_net = torch.load(model_dir)
        if use_cuda:
            R_net = R_net.cuda()
        R_net.eval()
        valid_f1, valid_exact = 0, 0
        prediction_dict={}
        if emb_input==1:
            valid_engine=valid_engine_1
        else:
            valid_engine=valid_engine_0
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
                    if j!=0 and args.data_version!=3:
                        sentense+=" "
                    if s_index>=e_index:
                        sentense +=  word_Vocab.get_word(dev_P[i,s_index-j])
                    else:
                        sentense +=  word_Vocab.get_word(dev_P[i,s_index+j])
                prediction_dict[dev_id_to_qid[dev_Q_id[i]]] = sentense

        print('valid_f1: %f | valid_exact: %f'%(
              valid_f1/len(valid_engine), valid_exact/len(valid_engine)))



        with open(Predict_base+r'/'+args.output_name+'_for_char_input'+str(char_input)+'_emb_input_'+str(emb_input)+'_concat_'+str(concat)+'_hidden_'+str(hidden)+'_batch_size_'+str(batch_size)+'_'+str(mode)+'_epoch_'+str(epoch)+'.json','w') as f:
            json.dump(prediction_dict,f)
