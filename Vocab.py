import numpy as np
import pickle

class Vocab_class:
    def __init__(self,w_to_i=None,i_to_w=None):
        self.w_to_i = w_to_i
        self.i_to_w = i_to_w
        if w_to_i is not None and i_to_w is not None :
            self.PAD_ID = self.w_to_i['--PAD--']
            self.UNK_ID = self.w_to_i['--OOV--']
    def save_dict(self,dict_obj, name ):
        with open( name + '.pkl', 'wb') as f:
            pickle.dump(dict_obj, f)
    def save(self,w2i_dir,i2w_dir):
        self.save_dict(self.w_to_i,w2i_dir)
        self.save_dict(self.i_to_w,i2w_dir)
    def load_dict(self,name):
        with open(name+'.pkl','rb') as f:
            return pickle.load(f)
    def load(self,w2i_dir,i2w_dir):
        self.w_to_i = self.load_dict(w2i_dir)
        self.i_to_w = self.load_dict(i2w_dir)
        self.PAD_ID = self.w_to_i['--PAD--']
        self.UNK_ID = self.w_to_i['--OOV--']
    def create_padded_list(self, list_of_py_arrays, max_len,pad_value=None):
        input_len=len(list_of_py_arrays)
        if pad_value is None:
            pad_value=self.PAD_ID
        if max_len >= input_len:
            return list_of_py_arrays + [pad_value] * (max_len - len(list_of_py_arrays))
        else:
            return list_of_py_arrays[:max_len] 
    def get_id(self,word):
        if word in self.w_to_i:
            return self.w_to_i[word]
        else:
            return self.w_to_i['--OOV--']
    def get_word(self,index):
        if index in self.i_to_w:
            return self.i_to_w[index]
        if index != self.UNK_ID and index != self.PAD_ID:
            print('Input Word index is not define in this Vocab class')
            print('So this Word define to OOV return OOV')
            return self.i_to_w[self.UNK_ID]