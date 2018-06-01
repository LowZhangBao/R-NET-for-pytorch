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
from Vocab import Vocab_SQUAD 
from module import R_Net,decode
from metrics import batch_score
use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Model and Training parameters')
# Model Architecture
parser.add_argument('--hidden_size', type=int, default=75, help='the hidden size of RNNs [75]')
# Training hyperparameter
parser.add_argument('--dev_phase',type=int,default=1,help='every epoch calculate dev score?')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=32, help='the size of batch [32]')
parser.add_argument('--lr', type=float, default=1, help='the learning rate of encoder [1]')
parser.add_argument('--display_freq', type=int, default=1, help='display training status every N iters [10]')
parser.add_argument('--save_freq', type=int, default=1, help='save model every N epochs [1]')
parser.add_argument('--save_batch_freq',type=int,default=500,help='save model every M batchs[500]')
parser.add_argument('--epoch', type=int, default=35, help='train for N epochs [25]')
parser.add_argument('--encoder_concat',type=int,default=1,help='use encoder concat')
parser.add_argument('--seed',type=int,default=1023,help='random_seed')
parser.add_argument('--ii_iter',type=int,default=1,help='repeat training batch n iter ')
args = parser.parse_args()

if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    train_P_dir = r"./SQUAD/train/train_P.npy"
    train_Q_dir = r"./SQUAD/train/train_Q.npy"
    train_A_dir = r"./SQUAD/train/train_A.npy"

    dev_P_dir = r"./SQUAD/dev/dev_P.npy"
    dev_Q_dir = r"./SQUAD/dev/dev_Q.npy"
    dev_A_dir = r"./SQUAD/dev/dev_A.npy"
    Embedding_output_dir=r"./SQUAD/squad_word_embedding.npy"

    train_P = np.load(train_P_dir)
    train_Q = np.load(train_Q_dir)
    train_A = np.load(train_A_dir).astype(np.float32)
    dev_P = np.load(dev_P_dir)
    dev_Q = np.load(dev_Q_dir)
    dev_A = np.load(dev_A_dir).astype(np.float32)
    train_P_mask=np.zeros(train_P.shape,dtype=np.uint8) 
    train_Q_mask=np.zeros(train_Q.shape,dtype=np.uint8)
    dev_P_mask=np.zeros(dev_P.shape,dtype=np.uint8)
    dev_Q_mask=np.zeros(dev_Q.shape,dtype=np.uint8)

    SQUAD_Vocab = Vocab_SQUAD()
    SQUAD_Vocab.load()
    PAD_ID = SQUAD_Vocab.PAD_ID
    UNK_ID = SQUAD_Vocab.UNK_ID


    train_P_mask[train_P==PAD_ID] = 1
    train_P_mask[train_P==UNK_ID] = 0 
    train_Q_mask[train_Q==PAD_ID] = 1
    train_Q_mask[train_Q==UNK_ID] = 0
    dev_P_mask[dev_P==PAD_ID] = 1
    dev_P_mask[dev_P==UNK_ID] = 0
    dev_Q_mask[dev_Q==PAD_ID] = 1
    dev_Q_mask[dev_Q==UNK_ID] = 0    
    embedding=np.load(Embedding_output_dir)
    

    
    print('Vocab size: %d | Max context: %d | Max question: %d'%(
          embedding.shape[0], train_P.shape[1], train_Q.shape[1]))

    print('Train: %d | Valid: %d | Test: %d'%(
          train_Q.shape[0],0,dev_P.shape[0] ))


    train_engine = DataLoader(data.DataEngine(train_P,
                                              train_Q,
                                              train_A,
                                              train_P_mask,
                                              train_Q_mask,
                                              embedding),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=use_cuda)
    
    valid_engine = DataLoader(data.DataEngine(dev_P,
                                              dev_Q,
                                              dev_A,
                                              dev_P_mask,
                                              dev_Q_mask,
                                              embedding),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=use_cuda)


    R_net = R_Net(batch_size=args.batch_size,
                  embedding_size=300,
                  hidden_size=75,
                  dropout=args.dropout,
                  encoder_concat=args.encoder_concat
                  )
    if use_cuda:
        R_net = R_net.cuda()
        print('use_cuda!  ')
    criterion = nn.NLLLoss()
    optimizer = optim.Adadelta(R_net.parameters(),lr=args.lr,rho=0.95,eps=1e-06)
    start_time= datetime.datetime.now()
    for epoch in range(args.epoch):
        batch = 0
        R_net.train()
        for p, q, ans_offset,p_mask,q_mask,idx in train_engine:
            p = Variable(p).cuda() if use_cuda else Variable(p)
            q = Variable(q).cuda() if use_cuda else Variable(q)
            p_mask = Variable(p_mask).cuda() if use_cuda else Variable(p_mask)
            q_mask = Variable(q_mask).cuda() if use_cuda else Variable(q_mask)
            start_ans = Variable(ans_offset[:, 0]).cuda() if use_cuda else Variable(ans_offset[:, 0])
            end_ans = Variable(ans_offset[:, 1]).cuda() if use_cuda else Variable(ans_offset[:, 1])
            for ii in range(args.ii_iter):
                batch_start = datetime.datetime.now()                           
                
                start1, start1_ori, end2, end2_ori = R_net(p,q,p_mask,q_mask)
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
                    torch.save(R_net, 'module'+'_now_epoch_'+str(epoch)+'now_batch_'+str(batch)+'_concat_'+str(args.encoder_concat)+'_batch_'+str(args.batch_size)+'_f1_'+str(f1_score)+'_em_'+str(exact_match_score)+'.cpt')

            batch +=1
        if args.dev_phase==True:
          valid_f1, valid_exact = 0, 0
          R_net.eval()
          for p, q, ans_offset,p_mask,q_mask,idx in valid_engine:
              p = Variable(p).cuda() if use_cuda else Variable(p)
              q = Variable(q).cuda() if use_cuda else Variable(q)
              p_mask = Variable(p_mask).cuda() if use_cuda else Variable(p_mask)
              q_mask = Variable(q_mask).cuda() if use_cuda else Variable(q_mask) 
              start_ans = Variable(ans_offset[:, 0]).cuda() if use_cuda else Variable(ans_offset[:, 0])
              end_ans = Variable(ans_offset[:,1]).cuda() if use_cuda else Variable(ans_offset[:, 1])
              
              start,_, end,_ = R_net(p,q,p_mask,q_mask)
              
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
              torch.save(R_net, 'module9'+'_now_epoch_'+str(epoch)+'_concat_'+str(args.encoder_concat)+'_batch_'+str(args.batch_size)+'_f1_'+str(vad_f1)+'_em_'+str(vad_em)+'.cpt')
    torch.save(R_net, 'module_final'+'_concat_'+str(args.encoder_concat)+'_f1_'+str(valid_f1)+'_em_'+str(valid_exact)+'.cpt')
    



