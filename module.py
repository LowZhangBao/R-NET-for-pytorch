import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,PackedSequence
import torch.nn.functional as F
import math
import numpy as np
import datetime

use_cuda = torch.cuda.is_available()

#############################################################################################################

def decode(score_s, score_e, top_n=1, max_len=None,mask_input=None):
    pred_s = []
    pred_e = []
    pred_score = []
    max_len = max_len or score_s.size(1)
    for i in range(score_s.size(0)):
        scores = torch.ger(score_s[i], score_e[i])
        scores.triu_().tril_(max_len - 1)
        scores = scores.numpy()
        scores_flat = scores.flatten()
        if top_n == 1:
            idx_sort = [np.argmax(scores_flat)]
        elif len(scores_flat) < top_n:
            idx_sort = np.argsort(-scores_flat)
        else:
            idx = np.argpartition(-scores_flat, top_n)[0:top_n]
            idx_sort = idx[np.argsort(-scores_flat[idx])]
        s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
        pred_s.append(s_idx)
        pred_e.append(e_idx)
        pred_score.append(scores_flat[idx_sort])
    return pred_s, pred_e, pred_score


class R_Net(nn.Module):
    def __init__(self,
                 char_embedding_size=300,
                 embedding_size=300,
                 hidden_size=75,
                 num_layers=3,
                 batch_size=1,
                 dropout=0.2,
                 encoder_concat=True,
                 version_flag=0,
                 char_input=True,
                 emb_input=True,
                 word_mat=None,
                 char_mat=None
                 ):

        super(R_Net, self).__init__()
        # --- R Net Structure --- #
        # Question And Passage Encoder
        self.version_flag   = version_flag
        self.character_dim  = char_embedding_size
        self.char_encoder_dim = 2*hidden_size
        self.question_dim   = embedding_size if char_input==False else embedding_size + self.char_encoder_dim
        self.context_dim    = embedding_size if char_input==False else embedding_size + self.char_encoder_dim
        self.batch          = batch_size
        self.hidden_size    = hidden_size
        self.attention_size = hidden_size
        self.encoder_concat = encoder_concat
        self.char_input = char_input
        self.emb_input = emb_input

        # Embedding matrix part
        if self.emb_input == False:
            self.char_emb = nn.Embedding(char_mat.shape[0],char_mat.shape[1])
            self.char_emb.weight.data.copy_(torch.from_numpy(char_mat))
            self.char_emb.weight.requires_grad=False
            self.word_emb = nn.Embedding(word_mat.shape[0],word_mat.shape[1])
            self.word_emb.weight.data.copy_(torch.from_numpy(word_mat))
            self.word_emb.weight.requires_grad=False
        # Char Encoder part
        self.P_char_gru = nn.GRU(input_size=self.character_dim,hidden_size=self.hidden_size,num_layers=1,bidirectional=True,batch_first=True)
        self.Q_char_gru = nn.GRU(input_size=self.character_dim,hidden_size=self.hidden_size,num_layers=1,bidirectional=True,batch_first=True)
        # Word Encoder part
        if encoder_concat ==True:
            self.Q_reader_gru1 = nn.GRU(input_size=self.question_dim ,hidden_size=self.hidden_size,num_layers=1,bidirectional=True,batch_first=True)
            self.Q_reader_gru2 = nn.GRU(input_size=self.hidden_size*2,hidden_size=self.hidden_size,num_layers=1,bidirectional=True,batch_first=True)
            self.Q_reader_gru3 = nn.GRU(input_size=self.hidden_size*2,hidden_size=self.hidden_size,num_layers=1,bidirectional=True,batch_first=True)
            self.P_reader_gru1 = nn.GRU(input_size=self.context_dim  ,hidden_size=self.hidden_size,num_layers=1,bidirectional=True,batch_first=True)
            self.P_reader_gru2 = nn.GRU(input_size=self.hidden_size*2,hidden_size=self.hidden_size,num_layers=1,bidirectional=True,batch_first=True)
            self.P_reader_gru3 = nn.GRU(input_size=self.hidden_size*2,hidden_size=self.hidden_size,num_layers=1,bidirectional=True,batch_first=True)
            self.reader_output = reader_output = self.hidden_size * 2 * 3
        else:
            self.Q_reader_gru = nn.GRU(input_size=question_dim      ,hidden_size=self.hidden_size,num_layers=num_layers,bidirectional=True,batch_first=True)
            self.P_reader_gru = nn.GRU(input_size=context_dim       ,hidden_size=self.hidden_size,num_layers=num_layers,bidirectional=True,batch_first=True)
            self.reader_output = reader_output = self.hidden_size *2
        # QP_Match_pack 
        self.QPP_gru_forward = nn.GRUCell(input_size = reader_output*2, hidden_size=self.hidden_size)
        self.QPP_gru_bakward = nn.GRUCell(input_size = reader_output*2, hidden_size=self.hidden_size)
        self.SM_gru_forward = nn.GRUCell(input_size=self.hidden_size*2, hidden_size=self.hidden_size)
        self.SM_gru_bakward = nn.GRUCell(input_size=self.hidden_size*2, hidden_size=self.hidden_size)
        self.PN_gru = nn.GRUCell(input_size=self.hidden_size*2, hidden_size=reader_output)

        self.v     = nn.Linear(self.hidden_size      ,                1    , bias=False)
        self.WQ_u  = nn.Linear(self.reader_output    , self.hidden_size    , bias=False)
        self.WP_u  = nn.Linear(self.reader_output    , self.hidden_size    , bias=False)
        self.WP_v  = nn.Linear(self.hidden_size      , self.hidden_size    , bias=False)
        self.W_g1  = nn.Linear(self.reader_output*2  , self.reader_output*2, bias=False)
        self.W_g2  = nn.Linear(self.hidden_size*2    , self.hidden_size*2  , bias=False)
        self.WP_h  = nn.Linear(self.hidden_size*2    , self.hidden_size    , bias=False)
        self.Wa_h  = nn.Linear(self.reader_output    , self.hidden_size    , bias=False)
        self.WQ_v  = nn.Linear(self.hidden_size*2    , self.hidden_size    , bias=False)
        self.WPP_v = nn.Linear(self.hidden_size      , self.hidden_size    , bias=False)
        self.VQ_r  = nn.Parameter(torch.randn(1,1,self.hidden_size*2),requires_grad=True)
        self.QPP_Wp_f = self.QPP_Wp_b = self.QPP_Wp = self.WP_u
        self.QPP_Wq_f = self.QPP_Wq_b = self.QPP_Wq = self.WQ_u
        self.QPP_Wv_f = self.QPP_Wv_b = self.QPP_Wv = self.WP_v
        self.QPP_Wg_f = self.QPP_Wg_b = self.QPP_Wg = self.W_g1
        self.QPP_V_f  = self.QPP_V_b  = self.QPP_V  = self.v
        self.SM_Wp_f  = self.SM_Wp_b  = self.SM_Wp  = self.WP_v
        self.SM_Wp__f = self.SM_Wp__b = self.SM_Wp_ = self.WPP_v
        self.SM_Wg_f  = self.SM_Wg_b  = self.SM_Wg  = self.W_g2
        self.SM_V_f   = self.SM_V_b   = self.SM_V   = self.v
        self.PN_VQr = torch.nn.Parameter(torch.randn(1,1,hidden_size*2),requires_grad=True)
        self.PN_gru = nn.GRUCell(input_size=self.hidden_size*2, hidden_size=reader_output)
        self.PN_WQv = self.WQ_v
        self.PN_WPh = self.WP_h
        self.PN_Wah = self.Wa_h
        self.PN_WQu = self.WQ_u
        self.PN_V   = self.v
    #new version for pac
    def new_char_encoder(self,char_input,gru_unit,char_mask,word_mask,max_len,batch):

        new_char_input=[]
        input_len=[]
        char_len=[]
        for i in range(batch):
            len_index = len(word_mask.data[i])-word_mask.data[i].sum()
            input_len.append(len_index)
            for j in range(len_index):
                char_index=len(char_mask.data[i,j])-char_mask.data[i,j,:].sum()
                char_len.append(char_index)
            new_char_input.append(char_input[i,:len_index])     
        char_len2 =np.asarray(char_len,dtype=np.int32)
        
        new_char_input = torch.cat(new_char_input,dim=0)
        
        sort_idx = np.argsort(-char_len2)
        new_char_input = new_char_input[sort_idx,:,:]
        char_len = char_len2[sort_idx]
        unsort_idx = np.argsort(sort_idx)
        
        pack_batch=new_char_input.size(0)
        
        new_char_input = pack_padded_sequence(new_char_input,char_len,batch_first=True)
        
        h = Variable(torch.rand(2*1, pack_batch,self.hidden_size))
        h = h.cuda() if use_cuda else h
        
        out_char,h_n = gru_unit(new_char_input,h)
        
        h_n = h_n.permute(1,0,2).contiguous().view(-1,self.hidden_size*2)
        h_n_ori = self.sort_tensor2D(h_n,unsort_idx)
        
        s_index=0
        out_h_n=[]
        for i in input_len:
            now_h_n=h_n_ori[s_index:s_index+i,:]
            s_index+=i
            now_h_n = F.pad(now_h_n,(0,0,0,max_len-int(i))).unsqueeze(0)
            out_h_n.append(now_h_n)
        out_h_n=torch.cat(out_h_n,dim=0)

        return out_h_n
    def Char_encoder(self,char_input,gru_unit,char_mask,word_mask,max_len,batch):

        input_len=[]
        batch_size=char_input.size(0)
        for i in range(batch):
            len_index = len(word_mask.data[i])-word_mask.data[i].sum()
            input_len.append(len_index)     
        input_len2=np.asarray(input_len,dtype=np.int32)
        output_array=[]
        for j in range(batch):
            batch_out=[]
            h = Variable(torch.zeros(2*1, 1,self.hidden_size ))
            h = h.cuda() if use_cuda else h
            char_len=[]
            for l in range(input_len2[j]):
                char_index=len(char_mask.data[j,l])-char_mask.data[j,l,:].sum()
                char_len.append(char_index)
            temp_char=char_input[j,:input_len2[j]]     
            for l in range(input_len2[j]):
                temp_input = temp_char[l,:char_len[l]].unsqueeze(0)
                temp2_input,h_n = gru_unit(temp_input,h)
                h_n=h_n.view(-1).unsqueeze(0)
                batch_out.append(h_n)
            batch_out = torch.cat(batch_out,dim=0).unsqueeze(0)
            batch_out = F.pad(batch_out,(0,0,0,max_len-int(input_len[j])))
            output_array.append(batch_out)

        output_array = torch.cat(output_array,dim=0)

        return output_array
    def Reader_encoder(self,input,gru_unit_list,batch):
        if self.encoder_concat == True:
            h = Variable(torch.zeros(2*1, batch,self.hidden_size ))
        else:
            h = Variable(torch.zeros(2*3,batch,self.hidden_size))
        h = h.cuda() if use_cuda else h
        temp_output=[]

        for gru_unit in gru_unit_list:
            input , h_n = gru_unit(input,h)
            temp_output.append(input)   
        
        return temp_output
    def Encoder_refix(self,input_pack,unsort_index):
        unpack_data = None
        
        for i in input_pack:
            pack_data , pack_len = pad_packed_sequence(i,batch_first=True)
            pack_data = pack_data[unsort_index,:,:]
            if isinstance(unpack_data,Variable):
                unpack_data = torch.cat((unpack_data,pack_data),dim=2)
            else:
                unpack_data = pack_data
        unpack_data = nn.Dropout(0.2)(unpack_data)
        return unpack_data
    def Encoder_pack(self,input_pack):
        concat_pack_data = None
        
        for i in input_pack:
            pack_data , pack_len = pad_packed_sequence(i,batch_first=True)
            #pack_data = pack_data[unsort_index,:,:]
            if isinstance(concat_pack_data,Variable):
                concat_pack_data = torch.cat((concat_pack_data,pack_data),dim=2)
            else:
                concat_pack_data = pack_data

        concat_pack_data = nn.Dropout(0.2)(concat_pack_data)
        concat_pack = pack_padded_sequence(concat_pack_data,pack_len,batch_first=True)
        return concat_pack
    def QP_Match_pack_fb(self,uP,uQ,Q_mask,use_Gate=False):
        uP_data = uP.data              
        uP_batch = uP.batch_sizes       
        ########################################
        #forward part
        ########################################
        vP_forward = []
        input_index = 0

        temp_batch_size = max_batch_size = int(uP_batch[0])
        temp_f_hidden = Variable(torch.zeros(max_batch_size,self.hidden_size))
        temp_f_hidden = temp_f_hidden.cuda() if use_cuda else temp_f_hidden
        now_hidden = temp_f_hidden
        for batch_size in uP_batch:
            step_uP = uP_data[input_index:input_index + batch_size,:]   # (2,450)
            input_index += batch_size                                   # (0,2,4,6,...)
            diff = temp_batch_size - batch_size
            if diff > 0:
                now_hidden = now_hidden[:-diff,:]
            temp_batch_size = batch_size
            #calculate part
            WuP = self.QPP_Wp_f(step_uP).unsqueeze(1)                                     
            Wuq = self.QPP_Wq_f(uQ[:batch_size,:,:])                             
            Wvv = self.QPP_Wv_f(now_hidden.squeeze(1)).unsqueeze(1)                 
            x = F.tanh(WuP + Wuq + Wvv)                                          
            s = self.QPP_V_f(x).squeeze(2)                                               
            Q_mask1 = Q_mask[:batch_size,:]
            s.data.masked_fill_(Q_mask1.data,-1e12)
            a = F.softmax(s,1).unsqueeze(1)                                            
            c = torch.bmm(a,uQ[:batch_size,:,:]).squeeze(1)                                             
            r = torch.cat([step_uP,c],dim=1)                 
            
            if use_Gate ==False:
                now_hidden = self.QPP_gru_forward(r, now_hidden)
            else:
                g = F.sigmoid(self.QPP_Wg_f(r))                        
                r = torch.mul(g, r)                       
                now_hidden = self.QPP_gru_forward(r,now_hidden)             
            vP_forward.append(now_hidden)
        vP_forward = torch.cat(vP_forward,dim=0)
        
        ########################################
        #backward part
        ########################################
        '''
        vP_bacward = []
        input_index = uP_data.size(0)

        temp_batch_size = uP_batch[-1]
        temp_b_hidden = Variable(torch.zeros(max_batch_size,self.hidden_size))
        temp_b_hidden = temp_b_hidden.cuda() if use_cuda else temp_b_hidden
        original_b_hidden = temp_b_hidden
        now_hidden = temp_b_hidden[:temp_batch_size,:]
        for batch_size in reversed(uP_batch.detach().numpy()):
            step_uP = uP_data[input_index-batch_size:input_index,:]   
            input_index -= batch_size                                   
            diff = batch_size - temp_batch_size
            if diff > 0:
                now_hidden = torch.cat([now_hidden,original_b_hidden[temp_batch_size:batch_size,:]],dim=0)
            temp_batch_size = batch_size
            WuP = self.QPP_Wp_b(step_uP).unsqueeze(1)         
            Wuq = self.QPP_Wq_b(uQ[:batch_size,:,:])       
            Wvv = self.QPP_Wv_b(now_hidden.squeeze(1)).unsqueeze(1)     
            x = F.tanh(WuP + Wuq + Wvv)                                                 
            s = self.QPP_V_b(x).squeeze(2)             
            Q_mask2 = Q_mask[:batch_size,:]
            s.data.masked_fill_(Q_mask2.data,-1e12)
            a = F.softmax(s,1).unsqueeze(1)                                            
            c = torch.bmm(a,uQ[:batch_size,:,:]).squeeze(1)                                              
            r = torch.cat([step_uP,c],dim=1)                 
            if use_Gate ==False:
                now_hidden = self.QPP_gru_bakward(r, now_hidden)
            else:
                g = F.sigmoid(self.QPP_Wg_b(r))                        
                r = torch.mul(g, r)                                                                     
                now_hidden = self.QPP_gru_bakward(r,now_hidden)
                hidden_1 = now_hidden             
            vP_bacward.append(now_hidden)
        vP_bacward.reverse()
        vP_bacward = torch.cat(vP_bacward,dim=0)       
        vP = torch.cat([vP_forward,vP_bacward],dim=1)

        vP = PackedSequence(vP,uP_batch)
        '''
        vP = PackedSequence(vP_forward,uP_batch)
        return vP
    def Self_Match_pack_fb(self,vP,vP_unpack,P_mask,use_Gate=False):

        vP_data = vP.data               
        vP_batch = vP.batch_sizes       
        ########################################
        #forward part
        ########################################
        hP_forward = []
        input_index = 0

        temp_batch_size = max_batch_size = int(vP_batch[0])

        temp_f_hidden = Variable(torch.zeros(max_batch_size,self.hidden_size))
        temp_f_hidden = temp_f_hidden.cuda() if use_cuda else temp_f_hidden
        now_hidden = temp_f_hidden
        #now_hidden = temp_hidden[0,:batch_size,:]
        for batch_size in vP_batch:
            step_vP = vP_data[input_index:input_index + batch_size,:]   # (2,450)
            input_index += batch_size                                   

            diff = temp_batch_size - batch_size
            if diff > 0:
                now_hidden = now_hidden[:-diff,:]     

            temp_batch_size = batch_size

            WPv  = self.SM_Wp__f(step_vP).unsqueeze(1)                                     #(chang_batch,    1,hidden)
            WPPv = self.SM_Wp_f (vP_unpack[:batch_size,:,:])                              #(batch      ,P_len,hidden)
            
            x = F.tanh(WPv + WPPv)                                                       #(batch      ,Q_len,hidden)
            s = self.SM_V_f(x).squeeze(2)                                                  #(batch      ,Q_len,1)==>(batch,Q_len)
            P_mask1 = P_mask[:batch_size,:]
            s.data.masked_fill_(P_mask1.data,-1e12)
            a = F.softmax(s,1).unsqueeze(1)                                              #(batch,    1,Q_len)
            c = torch.bmm(a,vP_unpack[:batch_size,:,:]).squeeze(1)                      #(batch,    1,Q_len)*(batch,Q_len,2*Q_dim)==>(batch,2*Q_dim)                            
            
            r = torch.cat([step_vP,c],dim=1)                 
            
            if use_Gate ==False:
                now_hidden = self.SM_gru_forward(r, now_hidden)
            else:
                g = F.sigmoid(self.SM_Wg_f(r))                        
                r = torch.mul(g, r)                                                
                now_hidden = self.SM_gru_forward(r,now_hidden)
                del g             
            hP_forward.append(now_hidden)
        del step_vP,input_index,diff,now_hidden,temp_batch_size,temp_f_hidden
        del WPv,WPPv,x,s,P_mask1,a,c,r
        hP_forward = torch.cat(hP_forward,dim=0)
        
        ########################################
        #backward part
        ########################################
        hP_bacward = []
        input_index = vP_data.size(0)

        temp_batch_size = int(vP_batch[-1])
        temp_b_hidden = Variable(torch.zeros(max_batch_size,self.hidden_size))
        temp_b_hidden = temp_b_hidden.cuda() if use_cuda else temp_b_hidden
        original_b_hidden = temp_b_hidden
        now_hidden = temp_b_hidden[:temp_batch_size,:]

        if isinstance(vP_batch,list):
            vP_batch_fix=vP_batch
        else:
            vP_batch_fix=vP_batch.detach().numpy()
            

        for batch_size in reversed(vP_batch_fix):
            step_vP = vP_data[input_index-batch_size:input_index,:]   # (2,450)
            
            diff = batch_size - temp_batch_size
            if diff > 0:
                now_hidden = torch.cat([now_hidden,original_b_hidden[temp_batch_size:batch_size,:]],dim=0)


            input_index -= batch_size                                   # (0,2,4,6,...)
            
            temp_batch_size = batch_size

            WPv  = self.SM_Wp__b(step_vP).unsqueeze(1)                                     #(chang_batch,    1,hidden)
            WPPv = self.SM_Wp_b (vP_unpack[:batch_size,:,:])                              #(batch      ,P_len,hidden)
            
            x = F.tanh(WPv + WPPv)                                                       #(batch      ,Q_len,hidden)
            s = self.SM_V_b(x).squeeze(2)                                                  #(batch      ,Q_len,1)==>(batch,Q_len)
            P_mask2 = P_mask[:batch_size,:]
            s.data.masked_fill_(P_mask2.data,-1e12)
            a = F.softmax(s,1).unsqueeze(1)                                              #(batch,    1,Q_len)
            c = torch.bmm(a,vP_unpack[:batch_size,:,:]).squeeze(1)                                              #(batch,    1,Q_len)*(batch,Q_len,2*Q_dim)==>(batch,2*Q_dim)                            
            
            r = torch.cat([step_vP,c],dim=1)                 
            
            if use_Gate ==False:
                now_hidden = self.SM_gru_bakward(r, now_hidden)
            else:
                g = F.sigmoid(self.SM_Wg_b(r))                        
                r = torch.mul(g, r)                                                
                now_hidden = self.SM_gru_bakward(r,now_hidden)
                del g             
            hP_bacward.append(now_hidden)
        hP_bacward.reverse()
        hP_bacward = torch.cat(hP_bacward,dim=0)       
        del step_vP,input_index,diff,now_hidden,temp_batch_size,temp_b_hidden,max_batch_size,original_b_hidden
        del WPv,WPPv,x,s,P_mask2,a,c,r       
        hP = torch.cat([hP_forward,hP_bacward],dim=1)
        #hP = nn.Dropout(0.2)(hP)
        hP = PackedSequence(hP,vP_batch)
        return hP
    def PointerNet_pack(self,hP_unpack,uQ,P_mask,Q_mask):

        WQu = self.PN_WQu(uQ)
        WQv = self.PN_WQv(self.PN_VQr)
        s   = self.PN_V(F.tanh(WQu+WQv)).squeeze(2)

        s.data.masked_fill_(Q_mask.data,-1e12)
        a   = F.softmax(s, 1).unsqueeze(1)
        rQ  = torch.bmm(a,uQ).squeeze(1)

        WPh = self.PN_WPh(hP_unpack)
        Wah = self.PN_Wah(rQ).unsqueeze(1)
        x   = F.tanh(WPh+Wah)
        s = self.PN_V(x).squeeze(2)
        s.data.masked_fill_(P_mask.data,-1e12)
        start_ori = s
        start = F.softmax(s,1)

        a = start.unsqueeze(1)
        c = torch.bmm(a, hP_unpack).squeeze(1)
        r = self.PN_gru(c,rQ)

        WPh = self.PN_WPh(hP_unpack)
        Wah = self.PN_Wah(r).unsqueeze(1)
        x   = F.tanh(WPh+Wah)
        s = self.PN_V(x).squeeze(2)
        s.data.masked_fill_(P_mask.data,-1e12)
        end_ori = s
        end = F.softmax(s, 1)
        start_ori = F.log_softmax(start_ori,1)
        end_ori = F.log_softmax(end_ori,1)


        return start,start_ori,end,end_ori,rQ
    def sort_tensor(self,input_vector,sort_idx):
        
        return input_vector[sort_idx,:,:]
    def sort_tensor2D(self,input_vector,sort_idx):
        
        return input_vector[sort_idx,:]
    def easy_pack(self,input_vector,input_len,sort_idx):
        input_vector = input_vector[sort_idx,:,:]
        input_vector = pack_padded_sequence(input_vector,input_len,batch_first=True)
        return input_vector
    def mask_abbr(self,input_mask,input_len):
        input_mask = input_mask[:,:input_len[0]]
        return input_mask
    def pack_input(self,input_vector,input_mask):
        input_len=[]
        batch_size=input_vector.size(0)
        for i in range(batch_size):
            len_index = len(input_mask.data[i])-input_mask.data[i].sum()
            input_len.append(len_index)
        input_len2=np.asarray(input_len,dtype=np.int32)
        
        sort_idx = np.argsort(-input_len2)
        input_vector = input_vector[sort_idx,:,:]
        input_len = input_len2[sort_idx]
        unsort_idx = np.argsort(sort_idx)

        input_vector = pack_padded_sequence(input_vector,input_len,batch_first=True)

        return input_vector,sort_idx,unsort_idx,input_len,input_len2
    def easy_unpack_without_unsort(self,pack):
        pack_data , pack_len = pad_packed_sequence(pack,batch_first=True)
        return pack_data,pack_len
    def forward(self,passage,question,p_mask,q_mask,pc,qc,pc_mask,qc_mask):
        self.batch = batch = passage.size(0)
        if self.emb_input == False:
            passage=self.word_emb(passage)
            question=self.word_emb(question)
        # Passage and Question char encoder

        if self.char_input==True:
            if self.emb_input == False:
                pc=self.char_emb(pc)
                qc=self.char_emb(qc)
            new_uQc = self.new_char_encoder(qc,self.Q_char_gru,qc_mask,q_mask,int(question.size(1)),batch)
            new_uPc = self.new_char_encoder(pc,self.P_char_gru,pc_mask,p_mask,int(passage.size(1)) ,batch)
            p = torch.cat([passage,new_uPc] ,dim=2)
            q = torch.cat([question,new_uQc],dim=2)
        else:
            p = passage
            q = question
        
        p_len = p.size(1)
        q_len = q.size(1)    
        p_pack_sort_p,p_sort_idx,p_unsort_idx,sort_p_len,real_p_len = self.pack_input(p,p_mask)
        q_pack_sort_q,q_sort_idx,q_unsort_idx,sort_q_len,real_q_len = self.pack_input(q,q_mask)
        p_mask = self.mask_abbr(p_mask,sort_p_len)
        q_mask = self.mask_abbr(q_mask,sort_q_len)
        
        # Passage and Question word encoder
        if self.encoder_concat == True:
            Q_gru_list=[self.Q_reader_gru1,self.Q_reader_gru2,self.Q_reader_gru3]
            P_gru_list=[self.P_reader_gru1,self.P_reader_gru2,self.P_reader_gru3]
        else:
            Q_gru_list=[self.Q_reader_gru]
            P_gru_list=[self.P_reader_gru]
        

        uQ_pack_list = self.Reader_encoder(q_pack_sort_q,Q_gru_list,batch)
        uP_pack_list = self.Reader_encoder(p_pack_sort_p,P_gru_list,batch)

        uQ_unpack_sort_original = self.Encoder_refix(uQ_pack_list,q_unsort_idx)
        uP_pack_sort_p          = self.Encoder_pack(uP_pack_list)
        q_mask_sort_p    = self.sort_tensor(q_mask.unsqueeze(2),p_sort_idx).squeeze(2)
        p_mask_sort_p    = self.sort_tensor(p_mask.unsqueeze(2),p_sort_idx).squeeze(2)    
        uQ_unpack_sort_p = self.sort_tensor(uQ_unpack_sort_original,p_sort_idx)
        

        vP_pack_sort_p   = self.QP_Match_pack_fb(uP_pack_sort_p,
                                                 uQ_unpack_sort_p,
                                                 q_mask_sort_p,
                                                 use_Gate=True)               
        torch.cuda.empty_cache()

        vP_unpack_sort_p , vP_len_sort_p = pad_packed_sequence(vP_pack_sort_p,batch_first=True)
        hP_pack_sort_p = self.Self_Match_pack_fb(vP_pack_sort_p,
                                                 vP_unpack_sort_p,
                                                 p_mask_sort_p,
                                                 use_Gate=True)
        torch.cuda.empty_cache()
        hP_unpack_sort_p, hP_len_sort_p = self.easy_unpack_without_unsort(hP_pack_sort_p)
        start,start_ori,end,end_ori,rQ = self.PointerNet_pack(hP_unpack_sort_p,
                                                              uQ_unpack_sort_p,
                                                              p_mask_sort_p,
                                                              q_mask_sort_p)
        torch.cuda.empty_cache()
        
        start       = self.sort_tensor2D(start    ,p_unsort_idx)
        start_ori   = self.sort_tensor2D(start_ori,p_unsort_idx)
        end         = self.sort_tensor2D(end      ,p_unsort_idx)
        end_ori     = self.sort_tensor2D(end_ori  ,p_unsort_idx)
        return start,start_ori,end,end_ori
