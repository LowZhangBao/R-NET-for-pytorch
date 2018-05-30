import numpy as np
import pickle

class Vocab_SQUAD:
    def __init__(self,w_to_i=None,i_to_w=None):
        self.w_to_i = w_to_i
        self.i_to_w = i_to_w
        if w_to_i is not None or i_to_w is not None :
            self.w_to_i['_PAD_']          =  len(self.w_to_i)
            self.i_to_w[len(self.w_to_i)-1] = '_PAD_'
            self.w_to_i['_UNK_']          = len(self.w_to_i)
            self.i_to_w[len(self.w_to_i)-1] = '_UNK_'
            self.PAD_ID = self.w_to_i['_PAD_']
            self.UNK_ID = self.w_to_i['_UNK_']
    def save_dict(self,dict_obj, name ):
        with open( name + '.pkl', 'wb') as f:
            pickle.dump(dict_obj, f, pickle.HIGHEST_PROTOCOL)
    def save(self):
        self.save_dict(self.w_to_i,'./SQUAD/SQUAD_Vocab_w_to_i')
        self.save_dict(self.i_to_w,'./SQUAD/SQUAD_Vocab_i_to_w')
    def load_dict(self,name):
        with open(name+'.pkl','rb') as f:
            return pickle.load(f)
    def load(self):
        self.w_to_i = self.load_dict('./SQUAD/SQUAD_Vocab_w_to_i')
        self.i_to_w = self.load_dict('./SQUAD/SQUAD_Vocab_i_to_w')
        self.PAD_ID = self.w_to_i['_PAD_']
        self.UNK_ID = self.w_to_i['_UNK_']
        
    def create_padded_list               (self, list_of_py_arrays, max_len):
        return list_of_py_arrays + [self.PAD_ID] * (max_len - len(list_of_py_arrays))  
    def create_padded_list_with_pad_value(self, list_of_py_arrays, max_len,pad_value):
        return list_of_py_arrays + [pad_value] * (max_len - len(list_of_py_arrays))
    def get_id(self,word):
        if word in self.w_to_i:
            return self.w_to_i[word]
        else:
            return self.UNK_ID
    def get_word(self,index):
        if index in self.i_to_w:
            return self.i_to_w[index]
        if index != self.UNK_ID and index != self.PAD_ID:
            print('Input Word is not define in this Vocab class')
            return None
        if index == self.UNK_ID:
            return "<UNIQUE_WORD>"
        if index == self.PAD_ID:
            return "<PADDING>"
class Vocab_char_SQUAD:
    def __init__(self,w_to_i=None,i_to_w=None):
        self.w_to_i = w_to_i
        self.i_to_w = i_to_w
    def save_dict(self,dict_obj, name ):
        with open( name + '.pkl', 'wb') as f:
            pickle.dump(dict_obj, f, pickle.HIGHEST_PROTOCOL)
    def save(self):
        self.save_dict(self.w_to_i,'./SQUAD/SQUAD_char_Vocab_w_to_i')
        self.save_dict(self.i_to_w,'./SQUAD/SQUAD_char_Vocab_i_to_w')
    def load_dict(self,name):
        with open(name+'.pkl','rb') as f:
            return pickle.load(f)
    def load(self):
        self.w_to_i = self.load_dict('./SQUAD/SQUAD_char_Vocab_w_to_i')
        self.i_to_w = self.load_dict('./SQUAD/SQUAD_char_Vocab_i_to_w')
        
    def create_padded_list               (self, list_of_py_arrays, max_len):
        return list_of_py_arrays + [self.PAD_ID] * (max_len - len(list_of_py_arrays))  
    def create_padded_list_with_pad_value(self, list_of_py_arrays, max_len,pad_value):
        return list_of_py_arrays + [pad_value] * (max_len - len(list_of_py_arrays))
    def get_id(self,word):
        if word in self.w_to_i:
            return self.w_to_i[word]
        else:
            return self.w_to_i['--OOV--']
    def get_word(self,index):
        if index in self.i_to_w:
            return self.i_to_w[index]
        if index != self.UNK_ID and index != self.PAD_ID:
            print('Input Word is not define in this Vocab class')
            return None