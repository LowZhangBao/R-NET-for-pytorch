import os
import jieba
import pickle
import json
import re
import sys
import math
import requests
import zipfile
import numpy as np
import setting
from Vocab import Vocab_class
from util import load_squad_file, load_glove_file_vocab,load_glove_to_squad_embedding,load_DRCD_file,load_fasttext_file_embedding
from util import create_char_array, create_char_vocab, create_floder,create_word_vocab
from util import get_spacy_list, get_line_count, get_embedding,get_line_count_fasttext,get_fasttext_embedding
from util import remove_blank,delete_numpy
from util import download_dataset

create_floder()
download_dataset()

train_dir=setting.DRCD_train_filename
dev_dir  =setting.DRCD_dev_filename
DRCD_dir=setting.DRCD_dir
DRCD_train_dir = setting.DRCD_train_dir
DRCD_dev_dir = setting.DRCD_dev_dir


train=load_DRCD_file(os.path.join(DRCD_dir,train_dir))
dev =load_DRCD_file( os.path.join(DRCD_dir,dev_dir))


print('preprocess DRCD data - train data!')
list_context       = []
list_context_char  = []

list_question      = []
list_question_char = []

id_to_qid={}
spans = []
q_id = np.arange(len(train['qids']))
for i in range(len(train['questions'])):
    id_to_qid[i] = train['qids'][i]
    now_question = train['questions'][i]
    now_question = remove_blank(now_question)
    q_token=jieba.cut(now_question,cut_all=False)
    q_token_text=[token for token in q_token]
    q_token_char=[[char for char in word]for word in q_token_text]
    now_passage = train['contexts'][train['qid2cid'][i]]
    now_passage = remove_blank(now_passage)
    p_token=jieba.cut(now_passage,cut_all=False)
    p_token_text=[token for token in p_token]
    p_token_char=[[char for char in word]for word in p_token_text]
    p_s_offset = {}
    p_e_offset = {}
    index=0
    for j in range(len(p_token_text)):
        token = p_token_text[j]
        p_s_offset[index]=token
        p_e_offset[index+len(token)-1]=token
        index+=len(token)
        
    now_answer = train['answers'][i]
    
    try:
        a_text = now_answer[0]['text']
        a_s = now_answer[0]['answer_start']
        a_e = a_s + len(a_text)-1
        token_s = token_e = None
        check_flag = a_s in p_s_offset and a_e in p_e_offset
        if check_flag == True:
            token_s_id = list(p_s_offset).index(a_s)
            token_e_id = list(p_e_offset).index(a_e)
        else:
            for j in range(len(p_token_text)):
                token = p_token_text[j]
                s = list(p_s_offset.keys())[j]
                e = list(p_e_offset.keys())[j]
                if s <= a_s and a_s <= e :
                    token_s = token
                    token_s_id = j
                    if j == len(token):
                        token_e = token
                elif token_s is not None:
                    token_e = token
                    token_e_id = j
                    if e >= a_e:
                        break
    except:
        token_s_id = None
        token_e_id = None
    
    
    list_context    .append (p_token_text)
    list_context_char.append(p_token_char)
    
    list_question    .append (q_token_text)
    list_question_char.append(q_token_char)
    spans.append([token_s_id,token_e_id])
    if i%1000 == 0 :
        print("Processed %d of %d (%f percent done)" % (i, len(train['questions']), 100 * float(i) / float(len(train['questions']))))

print('')
print('preprocess DRCD data - dev data!')
dev_list_context      = []
dev_list_context_char = []

dev_list_question     = []
dev_list_question_char= []

dev_id_to_qid={}
dev_spans = []
dev_q_id = np.arange(len(dev['qids']))
for i in range(len(dev['questions'])):
    dev_id_to_qid[i]=dev['qids'][i]
    now_question = dev['questions'][i]
    now_question = remove_blank(now_question)
    
    q_token=jieba.cut(now_question,cut_all=False)
    q_token_text=[token for token in q_token]
    q_token_char=[[char for char in word]for word in q_token_text]

        
    now_passage = dev['contexts'][dev['qid2cid'][i]]
    now_passage = remove_blank(now_passage)
    p_token=jieba.cut(now_passage,cut_all=False)
    p_token_text=[token for token in p_token]
    p_token_char=[[char for char in word]for word in p_token_text]
    p_s_offset = {}
    p_e_offset = {}
    index=0
    for j in range(len(p_token_text)):
        token = p_token_text[j]
        p_s_offset[index]=token
        p_e_offset[index+len(token)-1]=token
        index+=len(token)
        
    now_answer = train['answers'][i]
    
    try:
        a_text = now_answer[0]['text']
        a_s = now_answer[0]['answer_start']
        a_e = a_s + len(a_text)-1
        token_s = token_e = None
        check_flag = a_s in p_s_offset and a_e in p_e_offset
        if check_flag == True:
            token_s_id = list(p_s_offset).index(a_s)
            token_e_id = list(p_e_offset).index(a_e)
        else:
            for j in range(len(p_token_text)):
                token = p_token_text[j]
                s = list(p_s_offset.keys())[j]
                e = list(p_e_offset.keys())[j]
                if s <= a_s and a_s <= e :
                    token_s = token
                    token_s_id = j
                    if j == len(token):
                        token_e = token
                elif token_s is not None:
                    token_e = token
                    token_e_id = j
                    if e >= a_e:
                        break
    except:
        token_s_id = None
        token_e_id = None
    
    dev_list_context     .append(p_token_text)
    dev_list_context_char.append(p_token_char)
    dev_list_question     .append(q_token_text)
    dev_list_question_char.append(q_token_char)
    dev_spans.append([token_s_id,token_e_id])
    if i%500 == 0 :
        print("Processed %d of %d (%f percent done)" % (i, len(dev['questions']), 100 * float(i) / float(len(dev['questions']))))

print('Done!')

print('preprocess DRCD data - char phase!')
char_all_vocab_w2i={}
char_all_vocab_i2w={}
char_all_vocab_w2i["--OOV--"]=0
char_all_vocab_w2i["--PAD--"]=1
char_all_vocab_i2w[0]="--OOV--"
char_all_vocab_i2w[1]="--PAD--"
train_P_char_all,char_all_vocab_w2i,char_all_vocab_i2w = create_char_vocab(list_context_char     ,char_all_vocab_w2i,char_all_vocab_i2w,None,None)
train_Q_char_all,char_all_vocab_w2i,char_all_vocab_i2w = create_char_vocab(list_question_char    ,char_all_vocab_w2i,char_all_vocab_i2w,None,None)
dev_P_char_all,char_all_vocab_w2i,char_all_vocab_i2w   = create_char_vocab(dev_list_context_char ,char_all_vocab_w2i,char_all_vocab_i2w,None,None)
dev_Q_char_all,char_all_vocab_w2i,char_all_vocab_i2w   = create_char_vocab(dev_list_question_char,char_all_vocab_w2i,char_all_vocab_i2w,None,None)

char_Vocab_all = Vocab_class(char_all_vocab_w2i,char_all_vocab_i2w)
char_Vocab_all.save(os.path.join(DRCD_dir,setting.char_all_vocab_w2i_file) ,os.path.join(DRCD_dir,setting.char_all_vocab_i2w_file))



print('Done!')

print('')
print('dev  _context _max_len',np.max([len(dev_list_context[i]) for i in range(len(dev_list_context))]))
print('train_context _max_len',np.max([len(list_context[i]) for i in range(len(list_context))]))
print('dev  _question_max_len',np.max([len(dev_list_question[i]) for i in range(len(dev_list_question))]))
print('train_question_max_len',np.max([len(list_question[i]) for i in range(len(list_question))]))




w2i={}
i2w={}
w2i["--OOV--"]=0
w2i["--PAD--"]=1
i2w[0]="--OOV--"
i2w[1]="--PAD--"
print('create DRCD vocab word w2i and i2w')
w2i,i2w = create_word_vocab(list_context,w2i,i2w)
print("Done train passage")
w2i,i2w = create_word_vocab(list_question,w2i,i2w)
print("Done train question")
w2i,i2w = create_word_vocab(dev_list_context,w2i,i2w)
print("Done dev passage")
w2i,i2w = create_word_vocab(dev_list_question,w2i,i2w)
print("Done dev question")
word_Vocab = Vocab_class(w2i,i2w)
word_Vocab.save(os.path.join(DRCD_dir,setting.word_vocab_w2i_file),os.path.join(DRCD_dir,setting.word_vocab_i2w_file))

print('glove_vocab_w2i_len:',len(w2i))
print('glove_vocab_i2w_len:',len(i2w))

print('glove_vocab_char_all_w2i_len:'   ,len(char_all_vocab_w2i))
print('glove_vocab_char_all_i2w_len:'   ,len(char_all_vocab_i2w))







train_max_num   = len(list_context)
train_max_p_num = max([len(list_context[i])      for i in range(len(list_context     ))])
train_max_p_num = train_max_p_num if setting.dev_p_max == None else min(train_max_p_num,setting.dev_p_max)
train_max_q_num = max([len(list_question[i])     for i in range(len(list_question    ))])
train_max_q_num = train_max_q_num if setting.dev_q_max == None else min(train_max_q_num,setting.dev_q_max)

dev_max_num     = len(dev_list_context)
dev_max_p_num   = max([len(dev_list_context[i])  for i in range(len(dev_list_context ))])
dev_max_p_num   = dev_max_p_num if setting.dev_p_max == None else min(dev_max_p_num,setting.dev_p_max)
dev_max_q_num   = max([len(dev_list_question[i]) for i in range(len(dev_list_question))])
dev_max_q_num   = dev_max_q_num if setting.dev_q_max == None else min(dev_max_q_num,setting.dev_q_max)


print('create training data array')
print('create training data array')
train_P     = np.zeros((train_max_num,train_max_p_num),dtype=np.uint32 )
train_Q     = np.zeros((train_max_num,train_max_q_num),dtype=np.uint32 )
train_A     = np.zeros((train_max_num,2              ),dtype=np.float32)
dev_P       = np.zeros((dev_max_num  ,dev_max_p_num  ),dtype=np.uint32 )
dev_Q       = np.zeros((dev_max_num  ,dev_max_q_num  ),dtype=np.uint32 )
dev_A       = np.zeros((dev_max_num  ,2              ),dtype=np.float32)
print('load data to training data _ train phase')
for i in range(train_max_num):
    temp_p = list_context[i]
    temp_p = [w2i[word] for word in temp_p]
    temp_p = np.array(word_Vocab.create_padded_list(temp_p,train_max_p_num),dtype=np.uint32)
    train_P[i,:] = temp_p
    temp_q = list_question[i]
    temp_q = [w2i[word] for word in temp_q]
    temp_q = np.array(word_Vocab.create_padded_list  (temp_q,train_max_q_num),dtype=np.uint32)
    train_Q[i,:] = temp_q
    train_A[i,:] = spans[i]
print('load data to training data _ dev phase')
for i in range(dev_max_num):
    temp_p = dev_list_context[i]
    temp_p = [w2i[word] for word in temp_p]
    temp_p = np.array(word_Vocab.create_padded_list(temp_p,dev_max_p_num),dtype=np.uint32)
    dev_P[i,:] = temp_p
    temp_q = dev_list_question[i]
    temp_q = [w2i[word] for word in temp_q]
    temp_q = np.array(word_Vocab.create_padded_list(temp_q,dev_max_q_num),dtype=np.uint32)
    dev_Q[i,:] = temp_q
    dev_A[i,:] = dev_spans[i]

print('save_training data')

np.save(os.path.join(DRCD_train_dir,setting.train_P_file)            ,train_P)
np.save(os.path.join(DRCD_train_dir,setting.train_Q_file)            ,train_Q)


np.save(os.path.join(DRCD_train_dir,setting.train_P_char_all_file)  ,train_P_char_all)
np.save(os.path.join(DRCD_train_dir,setting.train_Q_char_all_file)  ,train_Q_char_all)

np.save(os.path.join(DRCD_train_dir,setting.train_A_file)            ,train_A)

np.save(os.path.join(DRCD_train_dir,setting.train_Q_id_file)         ,q_id)

with open(os.path.join(DRCD_train_dir,setting.train_Q_id_to_qid_file), 'wb') as f:
    pickle.dump(id_to_qid, f)

np.save(os.path.join(DRCD_dev_dir,setting.dev_P_file)            ,dev_P)

np.save(os.path.join(DRCD_dev_dir,setting.dev_Q_file)            ,dev_Q)


np.save(os.path.join(DRCD_dev_dir,setting.dev_P_char_all_file)  ,dev_P_char_all)
np.save(os.path.join(DRCD_dev_dir,setting.dev_Q_char_all_file)  ,dev_Q_char_all)

np.save(os.path.join(DRCD_dev_dir,setting.dev_A_file)            ,dev_A)

np.save(os.path.join(DRCD_dev_dir,setting.dev_Q_id_file)         ,dev_q_id)

with open(os.path.join(DRCD_dev_dir,setting.dev_Q_id_to_qid_file), 'wb') as f:
    pickle.dump(dev_id_to_qid, f)


print('preprocess DRCD word data done !')

print('preprocess DRCD word data done !')



print('preprocess fasttext  word embedding data!')

DRCD_word_embedding,fast_word_embedding = get_fasttext_embedding(os.path.join(setting.FAST_dir,setting.fast_zh_filename),word_Vocab,embedding_init=setting.DRCD_word_init,out_dim=setting.word_dim)


np.save(os.path.join(DRCD_dir,setting.DRCD_Word_Embedding_file),DRCD_word_embedding)
np.save(setting.Fasttext_Word_Embedding_output_dir,fast_word_embedding)

print('preprocess GloVe data done!')#
print('preprocess DRCD data for  char embedding!')

DRCD_char_embedding ,fast_char_embedding= get_fasttext_embedding(os.path.join(setting.FAST_dir,setting.fast_zh_filename),char_Vocab_all   ,embedding_init=setting.DRCD_char_init,out_dim=setting.char_dim)


np.save(os.path.join(DRCD_dir,setting.DRCD_Char_Embedding_file)   ,DRCD_char_embedding)
np.save(setting.Fasttext_Char_Embedding_output_dir       ,fast_char_embedding)
print('preprocess SQUAD data for char embedding done!')