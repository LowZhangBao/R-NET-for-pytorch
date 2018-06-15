import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
import data
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
from Vocab import Vocab_class   
from util import load_squad_data,create_mask,get_data_engine,create_floder_dir
from module import R_Net,decode
from metrics import batch_score
import setting
use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Model and Training parameters')
# Model Architecture
parser.add_argument('--hidden_size', type=int, default=75, help='the hidden size of RNNs [75]')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--encoder_concat',type=int,default=1,help='use encoder concat')
parser.add_argument('--char_input',type=int,default=1,help='use char encoder input?')
parser.add_argument('--emb_input',type=int,default=1,help='emb input? use DataLoader create embedding representation?')

# Training hyperparameter
parser.add_argument('--dev_phase',type=int,default=1,help='every epoch calculate dev score?')
parser.add_argument('--epoch', type=int, default=15, help='train for N epochs [15]')
parser.add_argument('--batch_size', type=int, default=32, help='the size of batch [32]')
parser.add_argument('--batch_size_dev',type=int,default=16,help='the dev size of batch [16]')
parser.add_argument('--lr', type=float, default=1, help='the learning rate of encoder [1]')
parser.add_argument('--display_freq', type=int, default=1, help='display training status every N iters [1]')
parser.add_argument('--save_freq', type=int, default=1, help='save model every N epochs [1]')
parser.add_argument('--save_batch_freq',type=int,default=500,help='save model every M batchs[500]')
parser.add_argument('--seed',type=int,default=1023,help='random_seed')
parser.add_argument('--ii_iter',type=int,default=1,help='repeat training batch n iter ')
args = parser.parse_args()

if __name__ == '__main__':
    #Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    # create model_save_floder 
    create_dir=r'./Model_save/'+'module'+'_char_input_'+str(args.char_input)+'_concat_'+str(args.encoder_concat)+'_hidden_'+str(args.hidden_size)
    create_floder_dir(create_dir)

    # Load Squad Data
    train_P,train_Q,train_P_c,train_Q_c,train_A,dev_P,dev_Q,dev_P_c,dev_Q_c,dev_A = load_squad_data()
    
    #Get word Vocab and Embedding

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

    # Create Passage and question mask
    word_PAD_ID = word_Vocab.PAD_ID
    word_UNK_ID = word_Vocab.UNK_ID
    char_PAD_ID = char_Vocab.PAD_ID
    char_UNK_ID = char_Vocab.UNK_ID
  
    train_P_mask = create_mask(train_P,word_PAD_ID,word_UNK_ID)
    train_Q_mask = create_mask(train_Q,word_PAD_ID,word_UNK_ID)
    dev_P_mask   = create_mask(dev_P,word_PAD_ID,word_UNK_ID)
    dev_Q_mask   = create_mask(dev_Q,word_PAD_ID,word_UNK_ID)


    train_P_char_mask = create_mask(train_P_c,char_PAD_ID,char_UNK_ID)
    train_Q_char_mask = create_mask(train_Q_c,char_PAD_ID,char_UNK_ID)
    dev_P_char_mask   = create_mask(dev_P_c,char_PAD_ID,char_UNK_ID)
    dev_Q_char_mask   = create_mask(dev_Q_c,char_PAD_ID,char_UNK_ID)
    
    train_Q_id = np.load(setting.train_Q_id_dir)
    dev_Q_id = np.load(setting.dev_Q_id_dir)

    # Print information for Data

    print('Word Vocab size: %d | Char Vocab size: %d | Max context: %d | Max question: %d'%(
          word_embedding.shape[0],char_embedding.shape[0], train_P.shape[1], train_Q.shape[1]))

    print('Train: %d | Valid: %d | Test: %d'%(
          train_P.shape[0],0,dev_P.shape[0] ))
    
    # Get the Data Engine

    train_engine = get_data_engine(args.emb_input,train_P,train_Q,train_A,train_P_mask,train_Q_mask,train_P_c,train_Q_c,train_P_char_mask,train_Q_char_mask,train_Q_id,word_embedding,char_embedding,args.batch_size    ,use_cuda)
    valid_engine = get_data_engine(args.emb_input,  dev_P,  dev_Q,  dev_A,  dev_P_mask,  dev_Q_mask,  dev_P_c,  dev_Q_c,  dev_P_char_mask,  dev_Q_char_mask,  dev_Q_id,word_embedding,char_embedding,args.batch_size_dev,use_cuda)

    # Define Network

    R_net = R_Net(batch_size=args.batch_size,
                  char_embedding_size=setting.char_dim,
                  embedding_size=setting.word_dim,
                  hidden_size=75,
                  dropout=args.dropout,
                  encoder_concat=args.encoder_concat,
                  char_input=args.char_input,
                  emb_input=args.emb_input,
                  word_mat=word_embedding,
                  char_mat=char_embedding
                  )

    if use_cuda:
        R_net = R_net.cuda()
        print('use_cuda!  ')

    # Setting Loss and Optim.
    criterion = nn.NLLLoss()
    parameters = filter(lambda p : p.requires_grad,R_net.parameters())
    optimizer = optim.Adadelta(parameters,lr=args.lr,rho=0.95,eps=1e-06)

    start_time= datetime.datetime.now()
    for epoch in range(args.epoch):
        batch = 0
        R_net.train()
        for p, q, ans_offset,p_mask,q_mask,pc,qc,pc_mask,qc_mask,Q_ids,idx in train_engine:
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
            for ii in range(args.ii_iter):
                batch_start = datetime.datetime.now()                           
                start1, start1_ori, end2, end2_ori = R_net(p,q,p_mask,q_mask,pc,qc,pc_mask,qc_mask)
                
                loss1 = criterion(start1_ori, start_ans) 
                loss2 = criterion(end2_ori, end_ans)
                loss = loss1+loss2
                optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm(R_net.parameters(), 10)
                optimizer.step()
               
                start, end, scores = decode(start1.data.cpu(), end2.data.cpu(), 1)
                f1_score, exact_match_score = batch_score(start, end, ans_offset)
                
                batch_time = datetime.datetime.now() - batch_start
                end_time = datetime.datetime.now() - start_time

                if batch % args.display_freq == 0   :
                    print('epoch: %d | batch: %d/%d| loss1: %f | loss2: %f | f1: %f | exact: %f | batch_time: %s | total_time: %s'%
                        ( epoch, batch, len(train_engine), loss1.data[0],loss2.data[0],f1_score, exact_match_score,batch_time,end_time))

                if batch % args.save_batch_freq==0:
                    torch.save(R_net,create_dir+r'/train_epoch_'+str(epoch)+'_batch_'+str(batch)+'_f1_'+str(f1_score)[:4]+'_em_'+str(exact_match_score)[:4]+'.cpt')
                    #torch.save(R_net, 'module'+'_char_input_'+str(args.char_input)+'_epoch_'+str(epoch)+'_batch_'+str(batch)+'_concat_'+str(args.encoder_concat)+'_batch_'+str(args.batch_size)+'_f1_'+str(f1_score)+'_em_'+str(exact_match_score)+'.cpt')

            batch +=1
        if args.dev_phase==True:
            valid_f1, valid_exact = 0, 0
            R_net.eval()
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
                
                start,_, end,_ = R_net(p,q,p_mask,q_mask,pc,qc,pc_mask,qc_mask)
                
                start, end, scores = decode(start.data.cpu(), end.data.cpu(), 1)
                
                f1_score, exact_match_score = batch_score(start, end, ans_offset)
                valid_f1 += f1_score
                valid_exact += exact_match_score
                print('epoch: %d | valid_f1: %f | valid_exact: %f'%(
                      epoch, valid_f1/len(valid_engine), valid_exact/len(valid_engine)
                ))
            if epoch % args.save_freq == 0:
                vad_f1 = valid_f1/len(valid_engine)
                vad_em = valid_exact/len(valid_engine)
                torch.save(R_net,create_dir+r'/dev_epoch_'+str(epoch)+'_batch_'+str(batch)+'_f1_'+str(vad_f1)[:4]+'_em_'+str(vad_em)[:4]+'.cpt')

                #torch.save(R_net, 'module'+'_char_input_'+str(args.char_input)+'_epoch_'+str(epoch)+'_concat_'+str(args.encoder_concat)+'_batch_'+str(args.batch_size)+'_f1_'+str(vad_f1)+'_em_'+str(vad_em)+'.cpt')
        else:
            if epoch % args.save_freq == 0:
                torch.save(R_net,create_dir+r'/dev_epoch_'+str(epoch)+'_batch_'+str(batch)+'_f1_'+str(vad_f1)[:4]+'_em_'+str(vad_em)[:4]+'.cpt')
                #torch.save(R_net, 'module'+'_char_input_'+str(args.char_input)+'_epoch_'+str(epoch)+'_concat_'+str(args.encoder_concat)+'_batch_'+str(args.batch_size)+'.cpt')
    
    torch.save(R_net,create_dir+r'/final_epoch_'+str(epoch)+'_batch_'+str(batch)+'_f1_'+str(vad_f1)[:4]+'_em_'+str(vad_em)[:4]+'.cpt')

    #torch.save(R_net, 'module_final'+'_char_input_'+str(args.char_input)+'_concat_'+str(args.encoder_concat)+'_f1_'+str(valid_f1)+'_em_'+str(valid_exact)+'.cpt')
    



