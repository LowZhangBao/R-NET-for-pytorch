import os
import spacy
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
from util import load_squad_file, load_glove_file_vocab,load_glove_to_squad_embedding
from util import create_char_array, create_char_vocab, create_floder,create_word_vocab
from util import get_spacy_list, get_line_count, get_embedding
from util import remove_blank,delete_numpy
from util import download_dataset



create_floder()
download_dataset()



train=load_squad_file(setting.squad_data_train_dir)
dev =load_squad_file(setting.squad_data_dev_dir)
try:
  nlp=spacy.load('en')
except:
  nlp=spacy.load('en_core_web_sm')
ner_list,pos_list,pos_tag_list = get_spacy_list()

print('preprocess SQUAD data - train data!')

list_context       = []
list_context_char  = []
list_context_pos   = []
list_context_tag   = []
list_context_ner   = []
list_question      = []
list_question_char = []
list_question_pos  = []
list_question_tag  = []
list_question_ner  = []
id_to_qid={}
spans = []
q_id = np.arange(len(train['qids']))
for i in range(len(train['questions'])):
    id_to_qid[i] = train['qids'][i]
    now_question = train['questions'][i]
    now_question = remove_blank(now_question)
    q_token = nlp(now_question)
    q_token_text = [token.text for token in q_token]
    q_token_char = [[char for char in word   ] for word  in q_token_text]
    q_token_pos  = [pos_list    [token.pos_  ] for token in q_token     ]
    q_token_tag  = [pos_tag_list[token.tag_  ] for token in q_token     ]
    q_token_ner  = []
    for j in range(len(q_token_text)):
        q_token_ner.append(ner_list[q_token[j].ent_type_]) 
    

        
    now_passage = train['contexts'][train['qid2cid'][i]]
    now_passage = remove_blank(now_passage)
    p_token = nlp(now_passage)
    p_token_text = [token.text for token in p_token]
    p_token_char = [[char for char in word   ] for word  in p_token_text]
    p_token_pos  = [pos_list    [token.pos_  ] for token in p_token     ]
    p_token_tag  = [pos_tag_list[token.tag_  ] for token in p_token     ]
    p_token_ner  = []
    for j in range(len(p_token_text)):
        p_token_ner.append(ner_list[p_token[j].ent_type_])    
    
    p_s_offset = {}
    p_e_offset = {}
    for j in range(len(p_token)):
        token = p_token[j]
        p_s_offset[token.idx                  ] = token
        p_e_offset[token.idx + len(token.text)] = token
  
    now_answer = train['answers'][i]
    
    a_text = now_answer[0]['text']
    
    a_s = now_answer[0]['answer_start']
    a_e = a_s + len(a_text)
    
    token_s = token_e = None
    
    check_flag = a_s in p_s_offset and a_e in p_e_offset
    if check_flag is not True:
        for j in range(len(p_token)):
            token = p_token[j]
            s = token.idx
            e = s + len(token.text)
            if s <= a_s and a_s <= e :
                token_s = token
                if j == len(p_token) - 1:
                    token_e = token
            elif token_s is not None:
                token_e = token
                if e >= a_e:
                    break
    token_s    = token_s if token_s is not None else p_s_offset[a_s]
    token_e    = token_e if token_e is not None else p_e_offset[a_e]
    token_s_id = list(p_token).index(token_s)
    token_e_id = list(p_token).index(token_e)
    
    list_context    .append (p_token_text)
    list_context_pos.append (p_token_pos )
    list_context_tag.append (p_token_tag )
    list_context_ner.append (p_token_ner )
    list_context_char.append(p_token_char)
    
    list_question    .append (q_token_text)
    list_question_pos.append (q_token_pos )
    list_question_tag.append (q_token_tag )
    list_question_ner.append (q_token_ner )
    list_question_char.append(q_token_char)
    spans.append([token_s_id,token_e_id])
    if i%1000 == 0 :
        print("Processed %d of %d (%f percent done)" % (i, len(train['questions']), 100 * float(i) / float(len(train['questions']))))

print('')
print('preprocess SQUAD data - dev data!')
dev_list_context      = []
dev_list_context_char = []
dev_list_context_pos  = []
dev_list_context_tag  = []
dev_list_context_ner  = []
dev_list_question     = []
dev_list_question_char= []
dev_list_question_pos = []
dev_list_question_tag = []
dev_list_question_ner = []
dev_id_to_qid={}
dev_spans = []
dev_q_id = np.arange(len(dev['qids']))
for i in range(len(dev['questions'])):
    dev_id_to_qid[i]=dev['qids'][i]
    now_question = dev['questions'][i]
    now_question = remove_blank(now_question)
    q_token = nlp(now_question)
    q_token_text = [token.text for token in q_token]
    q_token_char = [[char for char in word   ] for word  in q_token_text]
    q_token_pos  = [pos_list    [token.pos_  ] for token in q_token     ]
    q_token_tag  = [pos_tag_list[token.tag_  ] for token in q_token     ]
    q_token_ner  = []
    for j in range(len(q_token_text)):
        q_token_ner.append(ner_list[q_token[j].ent_type_]) 
    

        
    now_passage = dev['contexts'][dev['qid2cid'][i]]
    now_passage = remove_blank(now_passage)
    p_token = nlp(now_passage)
    p_token_text = [token.text for token in p_token]
    p_token_char = [[char for char in word   ] for word  in p_token_text]
    p_token_pos  = [pos_list    [token.pos_  ] for token in p_token     ]
    p_token_tag  = [pos_tag_list[token.tag_  ] for token in p_token     ]
    p_token_ner  = []
    for j in range(len(p_token_text)):
        p_token_ner.append(ner_list[p_token[j].ent_type_])    
    
    p_s_offset = {}
    p_e_offset = {}
    for j in range(len(p_token)):
        token = p_token[j]
        p_s_offset[token.idx                  ] = token
        p_e_offset[token.idx + len(token.text)] = token
  
    now_answer = dev['answers'][i]
    
    a_text = now_answer[0]['text']
    
    a_s = now_answer[0]['answer_start']
    a_e = a_s + len(a_text)
    
    token_s = token_e = None
    
    check_flag = a_s in p_s_offset and a_e in p_e_offset
    if check_flag is not True:
        for j in range(len(p_token)):
            token = p_token[j]
            s = token.idx
            e = s + len(token.text)
            if s <= a_s and a_s <= e :
                token_s = token
                if j == len(p_token) - 1:
                    token_e = token
            elif token_s is not None:
                token_e = token
                if e >= a_e:
                    break
    token_s    = token_s if token_s is not None else p_s_offset[a_s]
    token_e    = token_e if token_e is not None else p_e_offset[a_e]
    token_s_id = list(p_token).index(token_s)
    token_e_id = list(p_token).index(token_e)
    

    
    dev_list_context     .append(p_token_text)
    dev_list_context_pos .append(p_token_pos )
    dev_list_context_tag .append(p_token_tag )
    dev_list_context_ner .append(p_token_ner )
    dev_list_context_char.append(p_token_char)
    
    dev_list_question     .append(q_token_text)
    dev_list_question_pos .append(q_token_pos )
    dev_list_question_tag .append(q_token_tag )
    dev_list_question_ner .append(q_token_ner )
    dev_list_question_char.append(q_token_char)
    dev_spans.append([token_s_id,token_e_id])
    if i%1000 == 0 :
        print("Processed %d of %d (%f percent done)" % (i, len(dev['questions']), 100 * float(i) / float(len(dev['questions']))))


print('preprocess SQUAD data - char simple phase!')

glove_char_length  = get_line_count(setting.glove_char_dir)
char_simple_vocab_w2i=load_glove_file_vocab(setting.glove_char_dir)
char_simple_vocab_w2i["--OOV--"]=len(char_simple_vocab_w2i)
char_simple_vocab_w2i["--PAD--"]=len(char_simple_vocab_w2i)
char_simple_vocab_i2w = {}

for index,word in enumerate(char_simple_vocab_w2i):
    char_simple_vocab_i2w[char_simple_vocab_w2i[word]]=word


c
char_Vocab_simple = Vocab_class(char_simple_vocab_w2i,char_simple_vocab_i2w)
char_Vocab_simple.save(setting.char_simple_vocab_w2i_dir,setting.char_simple_vocab_i2w_dir)



train_P_char_simple = create_char_array(list_context_char     ,char_simple_vocab_w2i,setting.train_p_max,setting.c_max)
train_Q_char_simple = create_char_array(list_question_char    ,char_simple_vocab_w2i,setting.train_q_max,setting.c_max)
dev_P_char_simple   = create_char_array(dev_list_context_char ,char_simple_vocab_w2i,setting.dev_p_max,setting.c_max)
dev_Q_char_simple   = create_char_array(dev_list_question_char,char_simple_vocab_w2i,setting.dev_q_max,setting.c_max)


print('Done!')

print('preprocess SQUAD data - char all phase!')
char_all_vocab_w2i={}
char_all_vocab_i2w={}
char_all_vocab_w2i["--OOV--"]=0
char_all_vocab_w2i["--PAD--"]=1
char_all_vocab_i2w[0]="--OOV--"
char_all_vocab_i2w[1]="--PAD--"
train_P_char_all,char_all_vocab_w2i,char_all_vocab_i2w = create_char_vocab(list_context_char     ,char_all_vocab_w2i,char_all_vocab_i2w,setting.train_p_max,setting.c_max)
train_Q_char_all,char_all_vocab_w2i,char_all_vocab_i2w = create_char_vocab(list_question_char    ,char_all_vocab_w2i,char_all_vocab_i2w,setting.train_q_max,setting.c_max)
dev_P_char_all,char_all_vocab_w2i,char_all_vocab_i2w   = create_char_vocab(dev_list_context_char ,char_all_vocab_w2i,char_all_vocab_i2w,setting.dev_p_max,setting.c_max)
dev_Q_char_all,char_all_vocab_w2i,char_all_vocab_i2w   = create_char_vocab(dev_list_question_char,char_all_vocab_w2i,char_all_vocab_i2w,setting.dev_q_max,setting.c_max)

char_Vocab_all = Vocab_class(char_all_vocab_w2i,char_all_vocab_i2w)
char_Vocab_all.save(setting.char_all_vocab_w2i_dir ,setting.char_all_vocab_i2w_dir)

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
print('create SQUAD vocab word w2i and i2w')
w2i,i2w = create_word_vocab(list_context,w2i,i2w)
print("Done train passage")
w2i,i2w = create_word_vocab(list_question,w2i,i2w)
print("Done train question")
w2i,i2w = create_word_vocab(dev_list_context,w2i,i2w)
print("Done dev passage")
w2i,i2w = create_word_vocab(dev_list_question,w2i,i2w)
print("Done dev question")
word_Vocab = Vocab_class(w2i,i2w)
word_Vocab.save(setting.word_vocab_w2i_dir,setting.word_vocab_i2w_dir)

print('glove_vocab_w2i_len:',len(w2i))
print('glove_vocab_i2w_len:',len(i2w))

print('glove_vocab_char_all_w2i_len:'   ,len(char_all_vocab_w2i))
print('glove_vocab_char_all_i2w_len:'   ,len(char_all_vocab_i2w))
print('glove_vocab_char_simple_w2i_len:',len(char_simple_vocab_w2i))
print('glove_vocab_char_simple_i2w_len:',len(char_simple_vocab_i2w))





train_max_num   = len(list_context)
train_max_p_num = max([len(list_context[i])      for i in range(len(list_context     ))])
train_max_p_num = train_max_p_num if setting.train_p_max == None else min(train_max_p_num,setting.train_p_max)
train_max_q_num = max([len(list_question[i])     for i in range(len(list_question    ))])
train_max_q_num = train_max_q_num if setting.train_q_max == None else min(train_max_q_num,setting.train_q_max)

dev_max_num     = len(dev_list_context)
dev_max_p_num   = max([len(dev_list_context[i])  for i in range(len(dev_list_context ))])
dev_max_p_num   = dev_max_p_num if setting.dev_p_max == None else min(dev_max_p_num,setting.dev_p_max)
dev_max_q_num   = max([len(dev_list_question[i]) for i in range(len(dev_list_question))])
dev_max_q_num   = dev_max_q_num if setting.dev_q_max == None else min(dev_max_q_num,setting.dev_q_max)

print('create training data array')
train_P     = np.zeros((train_max_num,train_max_p_num),dtype=np.uint32 )
train_P_pos = np.zeros((train_max_num,train_max_p_num),dtype=np.uint8 )
train_P_tag = np.zeros((train_max_num,train_max_p_num),dtype=np.uint8 )
train_P_ner = np.zeros((train_max_num,train_max_p_num),dtype=np.uint8 )
train_Q     = np.zeros((train_max_num,train_max_q_num),dtype=np.uint32 )
train_Q_pos = np.zeros((train_max_num,train_max_q_num),dtype=np.uint8 )
train_Q_tag = np.zeros((train_max_num,train_max_q_num),dtype=np.uint8 )
train_Q_ner = np.zeros((train_max_num,train_max_q_num),dtype=np.uint8 )
train_A     = np.zeros((train_max_num,2              ),dtype=np.float32)
dev_P       = np.zeros((dev_max_num  ,dev_max_p_num  ),dtype=np.uint32 )
dev_P_pos   = np.zeros((dev_max_num  ,dev_max_p_num  ),dtype=np.uint8 )
dev_P_tag   = np.zeros((dev_max_num  ,dev_max_p_num  ),dtype=np.uint8 )
dev_P_ner   = np.zeros((dev_max_num  ,dev_max_p_num  ),dtype=np.uint8 )
dev_Q       = np.zeros((dev_max_num  ,dev_max_q_num  ),dtype=np.uint32 )
dev_Q_pos   = np.zeros((dev_max_num  ,dev_max_q_num  ),dtype=np.uint8 )
dev_Q_tag   = np.zeros((dev_max_num  ,dev_max_q_num  ),dtype=np.uint8 )
dev_Q_ner   = np.zeros((dev_max_num  ,dev_max_q_num  ),dtype=np.uint8 )
dev_A       = np.zeros((dev_max_num  ,2              ),dtype=np.float32)

print('load data to training data _ train phase')
for i in range(train_max_num):
    temp_p = list_context[i]
    temp_p = [w2i[word] for word in temp_p]
    temp_p = np.array(word_Vocab.create_padded_list(temp_p,train_max_p_num),dtype=np.uint32)
    temp_pos = np.array(word_Vocab.create_padded_list(list_context_pos[i],train_max_p_num,pos_list['']       ),dtype=np.uint8)
    temp_tag = np.array(word_Vocab.create_padded_list(list_context_tag[i],train_max_p_num,pos_tag_list['NIL']),dtype=np.uint8) 
    temp_ner = np.array(word_Vocab.create_padded_list(list_context_ner[i],train_max_p_num,ner_list['']       ),dtype=np.uint8)
    train_P[i,:] = temp_p
    train_P_pos[i,:] = temp_pos
    train_P_tag[i,:] = temp_tag
    train_P_ner[i,:] = temp_ner
    temp_q = list_question[i]
    temp_q = [w2i[word] for word in temp_q]
    temp_q = np.array(word_Vocab.create_padded_list  (temp_q,train_max_q_num),dtype=np.uint32)
    temp_pos = np.array(word_Vocab.create_padded_list(list_question_pos[i],train_max_q_num,pos_list['']       ),dtype=np.uint8)
    temp_tag = np.array(word_Vocab.create_padded_list(list_question_tag[i],train_max_q_num,pos_tag_list['NIL']),dtype=np.uint8) 
    temp_ner = np.array(word_Vocab.create_padded_list(list_question_ner[i],train_max_q_num,ner_list['']       ),dtype=np.uint8)
    train_Q[i,:] = temp_q
    train_Q_pos[i,:] = temp_pos
    train_Q_tag[i,:] = temp_tag
    train_Q_ner[i,:] = temp_ner
    train_A[i,:] = spans[i]
print('load data to training data _ dev phase')
for i in range(dev_max_num):
    temp_p = dev_list_context[i]
    temp_p = [w2i[word] for word in temp_p]
    temp_p = np.array(Vocab.create_padded_list(temp_p,dev_max_p_num),dtype=np.uint32)
    temp_pos = np.array(Vocab.create_padded_list(dev_list_context_pos[i],dev_max_p_num,pos_list['']       ),dtype=np.uint8)
    temp_tag = np.array(Vocab.create_padded_list(dev_list_context_tag[i],dev_max_p_num,pos_tag_list['NIL']),dtype=np.uint8) 
    temp_ner = np.array(Vocab.create_padded_list(dev_list_context_ner[i],dev_max_p_num,ner_list['']       ),dtype=np.uint8)
    dev_P[i,:] = temp_p
    dev_P_pos[i,:] = temp_pos
    dev_P_tag[i,:] = temp_tag
    dev_P_ner[i,:] = temp_ner
    temp_q = dev_list_question[i]
    temp_q = [w2i[word] for word in temp_q]
    temp_q = np.array(Vocab.create_padded_list(temp_q,dev_max_q_num),dtype=np.uint32)
    temp_pos = np.array(Vocab.create_padded_list(dev_list_question_pos[i],dev_max_q_num,pos_list['']       ),dtype=np.uint8)
    temp_tag = np.array(Vocab.create_padded_list(dev_list_question_tag[i],dev_max_q_num,pos_tag_list['NIL']),dtype=np.uint8) 
    temp_ner = np.array(Vocab.create_padded_list(dev_list_question_ner[i],dev_max_q_num,ner_list['']       ),dtype=np.uint8)
    dev_Q[i,:] = temp_q
    dev_Q_pos[i,:] = temp_pos
    dev_Q_tag[i,:] = temp_tag
    dev_Q_ner[i,:] = temp_ner
    dev_A[i,:] = dev_spans[i]

#print("arrange data because max_passage_len ==> answers span not can't find this passage ==> delete this pair {Q,P,A}")
print("arrange data because max_passage_len ==> answers span not can't find this passage ==> delete this pair {Q,P,A}")
delete_number=[]
for i in range(len(spans)):
    a=spans[i][0]
    b=spans[i][1]
    if a >=setting.train_p_max or b>=setting.train_p_max:
        delete_number.append(i)
q_id=np.delete(q_id,delete_number)
temp_id_ = [id_to_qid.pop(ii,None) for ii in delete_number]

train_P             = delete_numpy(train_P,delete_number)
train_P_pos         = delete_numpy(train_P_pos,delete_number)
train_P_ner         = delete_numpy(train_P_ner,delete_number)
train_P_tag         = delete_numpy(train_P_tag,delete_number)
train_P_char_simple = delete_numpy(train_P_char_simple,delete_number)
train_P_char_all    = delete_numpy(train_P_char_all,delete_number)
train_Q             = delete_numpy(train_Q,delete_number)
train_Q_pos         = delete_numpy(train_Q_pos,delete_number)
train_Q_ner         = delete_numpy(train_Q_ner,delete_number)
train_Q_tag         = delete_numpy(train_Q_tag,delete_number)
train_Q_char_simple = delete_numpy(train_Q_char_simple,delete_number)
train_Q_char_all    = delete_numpy(train_Q_char_all,delete_number)
train_A             = delete_numpy(train_A,delete_number)


print('save_training data')




np.save('./SQUAD/train/train_P_char_simple.npy',train_P_char_simple)
np.save('./SQUAD/train/train_Q_char_simple.npy',train_Q_char_simple)
np.save('./SQUAD/dev/dev_P_char_simple.npy',dev_P_char_simple)
np.save('./SQUAD/dev/dev_Q_char_simple.npy',dev_Q_char_simple)
np.save('./SQUAD/train/train_P_char_all.npy',train_P_char_all)
np.save('./SQUAD/train/train_Q_char_all.npy',train_Q_char_all)
np.save('./SQUAD/dev/dev_P_char_all.npy',dev_P_char_all)
np.save('./SQUAD/dev/dev_Q_char_all.npy',dev_Q_char_all)
np.save('./SQUAD/train/train_P.npy'    ,train_P)
np.save('./SQUAD/train/train_P_pos.npy',train_P_pos)
np.save('./SQUAD/train/train_P_ner.npy',train_P_ner)
np.save('./SQUAD/train/train_P_tag.npy',train_P_tag)
np.save('./SQUAD/train/train_Q.npy'    ,train_Q)
np.save('./SQUAD/train/train_Q_pos.npy',train_Q_pos)
np.save('./SQUAD/train/train_Q_ner.npy',train_Q_ner)
np.save('./SQUAD/train/train_Q_tag.npy',train_Q_tag)
np.save('./SQUAD/train/train_Q_id.npy' ,q_id)
np.save('./SQUAD/train/train_A.npy'    ,train_A)
np.save('./SQUAD/dev/dev_P.npy'    ,dev_P)
np.save('./SQUAD/dev/dev_P_pos.npy',dev_P_pos)
np.save('./SQUAD/dev/dev_P_ner.npy',dev_P_ner)
np.save('./SQUAD/dev/dev_P_tag.npy',dev_P_tag)
np.save('./SQUAD/dev/dev_Q.npy'    ,dev_Q)
np.save('./SQUAD/dev/dev_Q_pos.npy',dev_Q_pos)
np.save('./SQUAD/dev/dev_Q_ner.npy',dev_Q_ner)
np.save('./SQUAD/dev/dev_Q_tag.npy',dev_Q_tag)
np.save('./SQUAD/dev/dev_Q_id.npy' ,dev_q_id)
np.save('./SQUAD/dev/dev_A.npy'    ,dev_A)

with open( './SQUAD/train/train_id_to_qid' + '.pkl', 'wb') as f:
    pickle.dump(id_to_qid, f)

with open( './SQUAD/dev/dev_id_to_qid' + '.pkl', 'wb') as f:
    pickle.dump(dev_id_to_qid, f)

print('preprocess SQUAD word data done !')



print('preprocess GloVe word embedding data!')

squad_word_embedding,glove_word_embedding = get_embedding(setting.glove_dir,word_Vocab,embedding_init=setting.word_embedding_init,out_dim=setting.word_dim)

np.save(setting.SQUAD_Word_Embedding_output_dir,squad_word_embedding)
np.save(setting.Glove_Word_Embedding_output_dir,glove_word_embedding)

print('preprocess GloVe data done!')#

print('preprocess SQUAD data for simple char embedding!')

squad_char_simple_embedding,glove_char_embedding = get_embedding(setting.glove_char_dir,char_Vocab_simple,embedding_init=setting.char_embedding_init,out_dim=setting.char_dim)
squad_char_all_embedding   ,_                    = get_embedding(setting.glove_char_dir,char_Vocab_all   ,embedding_init=setting.char_embedding_init,out_dim=setting.char_dim)


np.save(setting.SQUAD_Char_simple_Embedding_output_dir,squad_char_simple_embedding)
np.save(setting.SQUAD_Char_all_Embedding_output_dir   ,squad_char_all_embedding)
np.save(setting.Glove_Char_Embedding_output_dir       ,glove_char_embedding)
print('preprocess SQUAD data for char embedding done!')