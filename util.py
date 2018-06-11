import json
import re
import os
import sys
import math
import requests
import zipfile
import numpy as np
import setting
from Vocab import Vocab_class
def load_squad_file(file_name):
    data = json.load(open(file_name, 'r'))['data']
    output = {'qids': [], 'questions': [], 'answers': [],
              'contexts': [], 'qid2cid': []}
    for article in data:
        for paragraph in article['paragraphs']:
            output['contexts'].append(paragraph['context'])
            for qa in paragraph['qas']:
                output['qids'].append(qa['id'])
                output['questions'].append(qa['question'])
                output['qid2cid'].append(len(output['contexts']) - 1)
                output['answers'].append(qa['answers'])
    return output
def load_squad_data():
    train_P_dir      = setting.train_P_dir
    train_Q_dir      = setting.train_Q_dir
    train_P_char_dir = setting.train_P_char_all_dir if setting.use_all_char_vocab==True else setting.train_P_char_simple_dir 
    train_Q_char_dir = setting.train_Q_char_all_dir if setting.use_all_char_vocab==True else setting.train_Q_char_simple_dir
    train_A_dir      = setting.train_A_dir

    dev_P_dir        = setting.dev_P_dir
    dev_Q_dir        = setting.dev_Q_dir
    dev_P_char_dir   = setting.dev_P_char_all_dir if setting.use_all_char_vocab==True else setting.dev_P_char_simple_dir
    dev_Q_char_dir   = setting.dev_Q_char_all_dir if setting.use_all_char_vocab==True else setting.dev_Q_char_simple_dir
    dev_A_dir        = setting.dev_A_dir

    train_P   = np.load(train_P_dir)
    train_Q   = np.load(train_Q_dir)
    train_P_c = np.load(train_P_char_dir)
    train_Q_c = np.load(train_Q_char_dir)
    train_A   = np.load(train_A_dir).astype(np.float32)
    dev_P     = np.load(dev_P_dir)
    dev_Q     = np.load(dev_Q_dir)
    dev_P_c   = np.load(dev_P_char_dir)
    dev_Q_c   = np.load(dev_Q_char_dir)
    dev_A     = np.load(dev_A_dir).astype(np.float32)
    

    return train_P,train_Q,train_P_c,train_Q_c,train_A,dev_P,dev_Q,dev_P_c,dev_Q_c,dev_A
def load_glove_file_embedding(file_dir,embedding,max_dim=300):
    vocab = {}
    index=0
    try:
        f = open(file_dir,"r",encoding='utf-8')
    except:
        f = open(file_dir,"r")
    for line in f:
        idx = line.index(" ")
        word = line[:idx]
        word_embedding = line[idx+1:]
        vocab[word]=index
        embedding[index]=np.fromstring(word_embedding,dtype=np.float32,sep=' ')[:max_dim]
        index+=1
    
    f.close()
    return vocab,embedding
def load_glove_file_vocab(file_dir):
    vocab = {}
    try:
        f = open(file_dir,"r",encoding='utf-8')
    except:
        f = open(file_dir,"r")

    for line in f:
        idx = line.index(" ")
        word = line[:idx]
        vocab[word]=len(vocab) 
    f.close()
    return vocab
def load_glove_to_squad_embedding(glove_vocab,squad_vocab,glove_embedding,squad_embedding):  
    for i,now_word in enumerate(squad_vocab.w_to_i):
        squad_index=squad_vocab.w_to_i[now_word]
        if now_word in glove_vocab:
            glove_index = glove_vocab[now_word]
            squad_embedding[squad_index,:] = glove_embedding[glove_index,:]
    return squad_embedding

def create_char_array(char_list,char_vocab,word_max=None,char_max=None):
    _num=len(char_list)

    _word_num = max([len(char_list[i]) for i in range(len(char_list))]) 
    _word_num = _word_num if word_max == None else min(_word_num,word_max) 
    _char_num = max([max([len(char) for char in char_list[i]]) for i in range(len(char_list))])
    _char_num = _char_num if char_max == None else min(_char_num,char_max)
    print(_num,_word_num,_char_num)
    char_array = np.ones((_num,_word_num,_char_num),dtype=np.uint16)*char_vocab["--PAD--"]
    for i in range(_num):
        now_sentense = char_list[i]
        s_len = min(len(now_sentense),_word_num)
        for j in range(s_len):
            now_word = now_sentense[j]
            c_len = min(len(now_word),_char_num)
            for k in range(c_len):
                try:
                    char_array[i,j,k] = char_vocab[now_word[k]]
                except:
                    char_array[i,j,k] = char_vocab["--OOV--"]
    return char_array
def create_char_vocab(char_list,char_w2i,char_i2w,word_max=None,char_max=None):
    _num=len(char_list)
    _word_num= max([len(char_list[i]) for i in range(len(char_list))])
    _word_num = _word_num if word_max == None else min(_word_num,word_max) 
    _char_num=max([max([len(char) for char in char_list[i]]) for i in range(len(char_list))])
    _char_num = _char_num if char_max == None else min(_char_num,char_max)
    print(_num,_word_num,_char_num)
    char_array = np.ones((_num,_word_num,_char_num),dtype=np.uint16)*int(char_w2i["--PAD--"])
    for i in range(_num):
        now_sentense = char_list[i]
        s_len = min(len(now_sentense),_word_num)
        for j in range(s_len):
            now_word = now_sentense[j]
            c_len = min(len(now_word),_char_num)
            for k in range(c_len):
                now_char=now_word[k]
                if now_char not in char_w2i:
                    char_w2i[now_char]=len(char_w2i)
                    char_i2w[len(char_w2i)-1]=now_char
                char_array[i,j,k] = char_w2i[now_char]
    return char_array,char_w2i,char_i2w
def create_word_vocab(word_list,word_w2i,word_i2w):
    for i in range(len(word_list)):
      now_context=word_list[i]
      for j in range(len(now_context)):
          now_word=now_context[j]
          if now_word  not in word_w2i:
              word_w2i[now_word]=len(word_w2i)
              word_i2w[len(word_w2i)-1]=now_word
    return word_w2i,word_i2w
def remove_blank(input_str):
    input_str = input_str.lstrip()
    input_str = input_str.rstrip()
    return input_str
def create_floder():

    def check_floder(floder_dir):
        if not os.path.exists(floder_dir):
            os.mkdir(floder_dir)

    SQUAD_dir = './SQUAD'
    GLOVE_dir = './GLOVE'
    Train_dir = './SQUAD/train'
    DEV_dir   = './SQUAD/dev'
    TEMP_dir = './TEMP_DATA'
    check_floder(SQUAD_dir)
    check_floder(GLOVE_dir)
    check_floder(Train_dir)
    check_floder(DEV_dir)
    check_floder(TEMP_dir)
    print('create_floder_over')
def download_dataset():

    def download_for_url(filename,url,path):
        real_url  = os.path.join(url, filename)
        real_path = os.path.join(path,filename)
        if not os.path.exists(real_path):
            try:
                print("Downloading file {}...".format(filename))
                r = requests.get(real_url, stream=True)
                with open(real_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
            except AttributeError as e:
                print("Download error!")
                raise e
        else:
          print(filename,"is exists!")

    train_filename = "train-v1.1.json"
    dev_filename = "dev-v1.1.json"
    glove_char_filename="glove.840B.300d-char.txt"
    glove_zip = "glove.840B.300d.zip"
    glove_filename = "glove.840B.300d.txt"

    glove_url = "http://nlp.stanford.edu/data/"
    glove_char_url = "https://raw.githubusercontent.com/minimaxir/char-embeddings/master/"
    train_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
    dev_url  = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

    download_for_url(train_filename,train_url, './SQUAD/')

    download_for_url(dev_filename  ,dev_url  , './SQUAD/')

    download_for_url(glove_char_filename,glove_char_url,'./Glove')
    

    if not os.path.exists(os.path.join('./GloVe/',glove_filename)):
        download_for_url(glove_zip,glove_url, './GloVe/')

        zip_ref = zipfile.ZipFile(os.path.join('./GloVe/',glove_zip), 'r')
        zip_ref.extractall('./Glove/')
        zip_ref.close()
        os.remove(os.path.join('./GloVe/',glove_zip))
    else:
      print(glove_filename,"is exists!")
      print("all data download over!")
def get_embedding(glove_dir,squad_vocab,embedding_init=None,out_dim=300):

    glove_length = get_line_count(glove_dir)
    glove_embedding = np.zeros((glove_length,out_dim), dtype=np.float32)

    glove_vocab,glove_embedding = load_glove_file_embedding(glove_dir,glove_embedding,max_dim=out_dim)

    if embedding_init is None:
        squad_embedding = np.zeros((len(squad_vocab.w_to_i),out_dim),dtype=np.float32)
    else:
        squad_embedding = np.random.normal(scale=0.01,size=(len(squad_vocab.w_to_i),out_dim)) 

    squad_embedding = load_glove_to_squad_embedding(glove_vocab,squad_vocab,glove_embedding,squad_embedding)
    
    return squad_embedding,glove_embedding
def delete_numpy(input_array,delete_array):

    return np.delete(input_array,delete_array,0)
def get_spacy_list():
    ner_list={'PERSON':0,
          'NORP':1,
          'FACILITY':2,
          'ORG':3,
          'GPE':4,
          'LOC':5,
          'PRODUCT':6,
          'EVENT':7,
          'WORK_OF_ART':8,
          'LAW':9,
          'LANGUAGE':10,
          'DATE':11,
          'TIME':12,
          'PERCENT':13,
          'MONEY':14,
          'QUANTITY':15,
          'ORDINAL':16,
          'CARDINAL':17,
          '':18,
          'FAC':2}
    pos_list={'PUNCT':0,
          'SYM':1,
          'X':2,
          'ADJ':3,
          'VERB':4,
          'CCONJ':5,
          'NUM':6,
          'DET':7,
          'ADV':8,
          'ADP':9,
          'NOUN':10,
          'PROPN':11,
          'PART':12,
          'PRON':13,
          'SPACE':14,
          'INTJ':15,
          '':16}
    pos_tag_list={
          '-LRB-':0,
          '-RRB-':1,
          ',':2,
          ':':3,
          '.':4,
          "''":5,
          '""':6,
          "#":7,
          '``':8,
          '$':9,
          'ADD':10,
          'AFX':11,
          'BES':12,
          'CC':13,
          'CD':14,
          'DT':15,
          'EX':16,
          'FW':17,
          'GW':18,
          'HVS':19,
          'HYPH':20,
          'IN':21,
          'JJ':22,
          'JJR':23,
          'JJS':24,
          'LS':25,
          'MD':26,
          'NFP':27,
          'NIL':28,
          'NN':29,
          'NNP':30,
          'NNPS':31,
          'NNS':32,
          'PDT':33,
          'POS':34,
          'PRP':35,
          'PRP$':36,
          'RB':37,
          'RBR':38,
          'RBS':39,
          'RP':40,
          '_SP':41,
          'SYM':42,
          'TO':43,
          'UH':44,
          'VB':45,
          'VBD':46,
          'VBG':47,
          'VBN':48,
          'VBP':49,
          'VBZ':50,
          'WDT':51,
          'WP':52,
          'WP$':53,
          'WRB':54,
          'XX':55,
          '':56,}
    return ner_list,pos_list,pos_tag_list
def get_line_count(data_dir):
    temp_line=0
    try:
        f = open(data_dir,"r",encoding='utf-8')
    except:
        f = open(data_dir,"r")
    for _ in f:
        temp_line += 1
    print("Vocab size: %d" % temp_line)
    f.close()
    return temp_line
def create_mask(in_data,pad_id,unk_id):
    in_data_mask=np.zeros(in_data.shape,dtype=np.uint8) 
    in_data_mask[in_data==pad_id] = 1
    in_data_mask[in_data==unk_id] = 0
    return in_data_mask 