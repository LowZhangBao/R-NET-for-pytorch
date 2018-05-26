import pickle
import torch
from torch.utils.data import Dataset


class DataEngine(Dataset):
    def __init__(self,P,Q,A,P_mask,Q_mask,word_embedding):
        self.P=P
        self.Q=Q
        self.A=A        
        self.P_mask=P_mask
        self.Q_mask=Q_mask
        self.emb=word_embedding
    def __len__(self):
        return self.P.shape[0]

    def __getitem__(self, idx):
        return self.vectorize_for_squad(self.P[idx,:],
                                        self.Q[idx,:],
                                        self.A[idx,:],
                                        self.P_mask[idx,:],
                                        self.Q_mask[idx,:],
                                        idx)

    def vectorize_for_squad(self, P,Q,A,P_mask,Q_mask,idx):
        p = torch.FloatTensor(self.emb[P,:])
        q = torch.FloatTensor(self.emb[Q,:])
        p_m = torch.ByteTensor(P_mask)
        q_m = torch.ByteTensor(Q_mask)
        ans_offset = torch.LongTensor(A)
        return p, q, ans_offset,p_m,q_m,idx

class DataEngine_for_prediction(Dataset):
    def __init__(self,P,Q,A,P_mask,Q_mask,word_embedding,Q_ids):
        self.P=P
        self.Q=Q
        self.A=A        
        self.P_mask=P_mask
        self.Q_mask=Q_mask
        self.emb=word_embedding
        self.Q_ids=Q_ids
    def __len__(self):
        return self.P.shape[0]

    def __getitem__(self, idx):
        return self.vectorize_for_squad(self.P[idx,:],
                                        self.Q[idx,:],
                                        self.A[idx,:],
                                        self.P_mask[idx,:],
                                        self.Q_mask[idx,:],
                                        self.Q_ids[idx],
                                        idx)

    def vectorize_for_squad(self, P,Q,A,P_mask,Q_mask,Q_ids,idx):
        p = torch.FloatTensor(self.emb[P,:])
        q = torch.FloatTensor(self.emb[Q,:])
        p_m = torch.ByteTensor(P_mask)
        q_m = torch.ByteTensor(Q_mask)
        ans_offset = torch.LongTensor(A)
        return p, q, ans_offset,p_m,q_m,Q_ids,idx

