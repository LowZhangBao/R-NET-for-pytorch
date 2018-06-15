import pickle
import torch
from torch.utils.data import Dataset
import setting

class DataEngine(Dataset):
    def __init__(self,P,Q,A,P_mask,Q_mask,word_embedding,Q_ids,Pc=None,Qc=None,Pc_mask=None,Qc_mask=None,char_embedding=None):
        self.P=P
        self.Q=Q
        self.A=A        
        self.P_mask=P_mask
        self.Q_mask=Q_mask
        self.emb=word_embedding
        self.Q_ids=Q_ids

        if Pc is not None and Qc is not None :
            self.Pc=Pc
            self.Qc=Qc
            self.Pc_mask=Pc_mask
            self.Qc_mask=Qc_mask
            self.char_emb=char_embedding
    def __len__(self):
        return self.P.shape[0]

    def __getitem__(self, idx):
        if self.Pc is not None or self.Qc is not None:
            return self.vectorize_for_squad(self.P[idx,:],
                                            self.Q[idx,:],
                                            self.A[idx,:],
                                            self.P_mask[idx,:],
                                            self.Q_mask[idx,:],
                                            self.Pc[idx,:],
                                            self.Qc[idx,:],
                                            self.Pc_mask[idx,:],
                                            self.Qc_mask[idx,:],
                                            self.Q_ids[idx],
                                            idx)
        else:
            return self.vectorize_for_squad_word(self.P[idx,:],
                                                 self.Q[idx,:],
                                                 self.A[idx,:],
                                                 self.P_mask[idx,:],
                                                 self.Q_mask[idx,:],
                                                 self.Q_ids[idx],
                                                 idx)
    def vectorize_for_squad_word(self, P,Q,A,P_mask,Q_mask,Q_ids,idx):
        p = torch.FloatTensor(self.emb[P,:setting.word_dim])
        q = torch.FloatTensor(self.emb[Q,:setting.word_dim])
        p_m = torch.ByteTensor(P_mask)
        q_m = torch.ByteTensor(Q_mask)
        ans_offset = torch.LongTensor(A)
        _=None
        return p, q, ans_offset,p_m,q_m,_,_,_,_,Q_ids,idx
    def vectorize_for_squad(self, P,Q,A,P_mask,Q_mask,Pc,Qc,Pc_mask,Qc_mask,Q_ids,idx):
        p = torch.FloatTensor(self.emb[P,:setting.word_dim])
        q = torch.FloatTensor(self.emb[Q,:setting.word_dim])
        pc = torch.FloatTensor(self.char_emb[Pc,:setting.char_dim])
        qc = torch.FloatTensor(self.char_emb[Qc,:setting.char_dim])
        p_m = torch.ByteTensor(P_mask)
        q_m = torch.ByteTensor(Q_mask)
        pc_m = torch.ByteTensor(Pc_mask)
        qc_m = torch.ByteTensor(Qc_mask)
        ans_offset = torch.LongTensor(A)
        return p, q, ans_offset,p_m,q_m,pc,qc,pc_m,qc_m,Q_ids,idx

class DataEngine_no_emb(Dataset):
    def __init__(self,P,Q,A,P_mask,Q_mask,Q_ids,Pc=None,Qc=None,Pc_mask=None,Qc_mask=None):
        self.P=P
        self.Q=Q
        self.A=A        
        self.P_mask=P_mask
        self.Q_mask=Q_mask
        self.Q_ids=Q_ids

        if Pc is not None and Qc is not None :
            self.Pc=Pc
            self.Qc=Qc
            self.Pc_mask=Pc_mask
            self.Qc_mask=Qc_mask
    def __len__(self):
        return self.P.shape[0]

    def __getitem__(self, idx):
        if self.Pc is not None or self.Qc is not None:
            return self.vectorize_for_squad(self.P[idx,:],
                                            self.Q[idx,:],
                                            self.A[idx,:],
                                            self.P_mask[idx,:],
                                            self.Q_mask[idx,:],
                                            self.Pc[idx,:],
                                            self.Qc[idx,:],
                                            self.Pc_mask[idx,:],
                                            self.Qc_mask[idx,:],
                                            self.Q_ids[idx],
                                            idx)
        else:
            return self.vectorize_for_squad_word(self.P[idx,:],
                                                 self.Q[idx,:],
                                                 self.A[idx,:],
                                                 self.P_mask[idx,:],
                                                 self.Q_mask[idx,:],
                                                 self.Q_ids[idx],
                                                 idx)
    def vectorize_for_squad_word(self, P,Q,A,P_mask,Q_mask,Q_ids,idx):
        p = torch.LongTensor(P)
        q = torch.LongTensor(Q)
        p_m = torch.LongTensor(P_mask)
        q_m = torch.LongTensor(Q_mask)
        ans_offset = torch.LongTensor(A)
        _=None
        return p, q, ans_offset,p_m,q_m,_,_,_,_,Q_ids,idx
    def vectorize_for_squad(self, P,Q,A,P_mask,Q_mask,Pc,Qc,Pc_mask,Qc_mask,Q_ids,idx):
        p = torch.LongTensor(P)
        q = torch.LongTensor(Q)
        pc = torch.LongTensor(Pc)
        qc = torch.LongTensor(Qc)
        p_m = torch.ByteTensor(P_mask)
        q_m = torch.ByteTensor(Q_mask)
        pc_m = torch.ByteTensor(Pc_mask)
        qc_m = torch.ByteTensor(Qc_mask)
        ans_offset = torch.LongTensor(A)
        return p, q, ans_offset,p_m,q_m,pc,qc,pc_m,qc_m,Q_ids,idx

