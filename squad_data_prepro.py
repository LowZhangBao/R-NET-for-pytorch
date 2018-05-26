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
from Vocab import Vocab_SQUAD

def load_squad(file_name):
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

def get_line_count(data_dir):
    temp_line=0
    with open(data_dir, "r", encoding="utf-8") as f:
        for _ in f:
            temp_line += 1
        print("Vocab size: %d" % temp_line)
    return temp_line

def load_glove_vocab(file_dir,embedding):
    vocab = {}
    index=0
    with open(file_dir, encoding="utf8") as f:
        for line in f:
            idx = line.index(" ")
            word = line[:idx]
            word_embedding = line[idx+1:]
            vocab[word]=index
            embedding[index]=np.fromstring(word_embedding,dtype=np.float32,sep=' ')
            
            index+=1
            if index%10000 == 0 :
                print("Processed %d of %d (%f percent done)" % (index, glove_length, 100 * float(index) / float(glove_length)), end="\r")
    return vocab,embedding

def create_floder():

    def check_floder(floder_dir):
        if not os.path.exists(floder_dir):
            os.mkdir(floder_dir)

    SQUAD_dir = './SQUAD'
    GLOVE_dir = './GLOVE'
    Train_dir = './SQUAD/train'
    DEV_dir   = './SQUAD/dev'
    check_floder(SQUAD_dir)
    check_floder(GLOVE_dir)
    check_floder(Train_dir)
    check_floder(DEV_dir)
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



create_floder()
download_dataset()



squad_data_train_dir='./SQUAD/train-v1.1.json'
squad_data_dev_dir='./SQUAD/dev-v1.1.json'

glove_dir = r'./GloVe/glove.840B.300d.txt'

train=load_squad(squad_data_train_dir)
dev =load_squad(squad_data_dev_dir)
nlp=spacy.load('en_core_web_sm')

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
print('preprocess SQUAD data!')

list_context      = []
list_context_pos  = []
list_context_tag  = []
list_context_ner  = []
list_question     = []
list_question_pos = []
list_question_tag = []
list_question_ner = []
id_to_qid={}
spans = []
q_id = np.arange(len(train['qids']))
for i in range(len(train['questions'])):
    id_to_qid[i] = train['qids'][i]
    now_question = train['questions'][i]
    q_token = nlp(now_question)
    q_token_text = [token.text for token in q_token]
    q_token_pos  = [pos_list    [token.pos_  ] for token in q_token     ]
    q_token_tag  = [pos_tag_list[token.tag_  ] for token in q_token     ]
    q_token_ner  = []
    for j in range(len(q_token_text)):
        q_token_ner.append(ner_list[q_token[j].ent_type_]) 
    

        
    now_passage = train['contexts'][train['qid2cid'][i]]
    p_token = nlp(now_passage)
    p_token_text = [token.text for token in p_token]
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
    
    list_context    .append(p_token_text)
    list_context_pos.append(p_token_pos )
    list_context_tag.append(p_token_tag )
    list_context_ner.append(p_token_ner )
    
    list_question    .append(q_token_text)
    list_question_pos.append(q_token_pos )
    list_question_tag.append(q_token_tag )
    list_question_ner.append(q_token_ner )
    spans.append([token_s_id,token_e_id])
    if i%1000 == 0 :
        print("Processed %d of %d (%f percent done)" % (i, len(train['questions']), 100 * float(i) / float(len(train['questions']))), end="\r")

print('')

dev_list_context      = []
dev_list_context_pos  = []
dev_list_context_tag  = []
dev_list_context_ner  = []
dev_list_question     = []
dev_list_question_pos = []
dev_list_question_tag = []
dev_list_question_ner = []
dev_id_to_qid={}
dev_spans = []
dev_q_id = np.arange(len(dev['qids']))
for i in range(len(dev['questions'])):
    dev_id_to_qid[i]=dev['qids'][i]
    now_question = dev['questions'][i]
    q_token = nlp(now_question)
    q_token_text = [token.text for token in q_token]
    q_token_pos  = [pos_list    [token.pos_  ] for token in q_token     ]
    q_token_tag  = [pos_tag_list[token.tag_  ] for token in q_token     ]
    q_token_ner  = []
    for j in range(len(q_token_text)):
        q_token_ner.append(ner_list[q_token[j].ent_type_]) 
    

        
    now_passage = dev['contexts'][dev['qid2cid'][i]]
    p_token = nlp(now_passage)
    p_token_text = [token.text for token in p_token]
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
    

    
    dev_list_context    .append(p_token_text)
    dev_list_context_pos.append(p_token_pos )
    dev_list_context_tag.append(p_token_tag )
    dev_list_context_ner.append(p_token_ner )
    
    dev_list_question    .append(q_token_text)
    dev_list_question_pos.append(q_token_pos )
    dev_list_question_tag.append(q_token_tag )
    dev_list_question_ner.append(q_token_ner )
    dev_spans.append([token_s_id,token_e_id])
    if i%1000 == 0 :
        print("Processed %d of %d (%f percent done)" % (i, len(dev['questions']), 100 * float(i) / float(len(dev['questions']))), end="\r")

print('')
print('dev  _context _max_len',np.max([len(dev_list_context[i]) for i in range(len(dev_list_context))]))
print('train_context _max_len',np.max([len(list_context[i]) for i in range(len(list_context))]))
print('dev  _question_max_len',np.max([len(dev_list_question[i]) for i in range(len(dev_list_question))]))
print('train_question_max_len',np.max([len(list_question[i]) for i in range(len(list_question))]))



w2i={}
i2w={}

k=0
for i in range(len(list_context)):
    for j in range(len(list_context[i])):
        now_word=list_context[i][j]
        if now_word  not in w2i.keys():
            #print('yes')
            w2i[list_context[i][j]]=k
            i2w[k]=list_context[i][j]
            k+=1
for i in range(len(list_question)):
    for j in range(len(list_question[i])):
        now_word=list_question[i][j]
        if now_word  not in w2i.keys():
            #print('yes')
            w2i[list_question[i][j]]=k
            i2w[k]=list_question[i][j]
            k+=1   
for i in range(len(dev_list_context)):
    for j in range(len(dev_list_context[i])):
        now_word=dev_list_context[i][j]
        if now_word  not in w2i.keys():
            #print('yes')
            w2i[dev_list_context[i][j]]=k
            i2w[k]=dev_list_context[i][j]
            k+=1
for i in range(len(dev_list_question)):
    for j in range(len(dev_list_question[i])):
        now_word=dev_list_question[i][j]
        if now_word  not in w2i.keys():
            #print('yes')
            w2i[dev_list_question[i][j]]=k
            i2w[k]=dev_list_question[i][j]
            k+=1   
            


Vocab = Vocab_SQUAD(w2i,i2w)
Vocab.save()


train_max_num   = len(list_context)
train_max_p_num = max([len(list_context[i])      for i in range(len(list_context     ))])
train_max_q_num = max([len(list_question[i])     for i in range(len(list_question    ))])
dev_max_num     = len(dev_list_context)
dev_max_p_num   = max([len(dev_list_context[i])  for i in range(len(dev_list_context ))])
dev_max_q_num   = max([len(dev_list_question[i]) for i in range(len(dev_list_question))])


train_P     = np.zeros((train_max_num,train_max_p_num),dtype=np.uint32 )
train_P_pos = np.zeros((train_max_num,train_max_p_num),dtype=np.uint32 )
train_P_tag = np.zeros((train_max_num,train_max_p_num),dtype=np.uint32 )
train_P_ner = np.zeros((train_max_num,train_max_p_num),dtype=np.uint32 )
train_Q     = np.zeros((train_max_num,train_max_q_num),dtype=np.uint32 )
train_Q_pos = np.zeros((train_max_num,train_max_q_num),dtype=np.uint32 )
train_Q_tag = np.zeros((train_max_num,train_max_q_num),dtype=np.uint32 )
train_Q_ner = np.zeros((train_max_num,train_max_q_num),dtype=np.uint32 )
train_A     = np.zeros((train_max_num,2              ),dtype=np.float32)
dev_P       = np.zeros((dev_max_num  ,dev_max_p_num  ),dtype=np.uint32 )
dev_P_pos   = np.zeros((dev_max_num  ,dev_max_p_num  ),dtype=np.uint32 )
dev_P_tag   = np.zeros((dev_max_num  ,dev_max_p_num  ),dtype=np.uint32 )
dev_P_ner   = np.zeros((dev_max_num  ,dev_max_p_num  ),dtype=np.uint32 )
dev_Q       = np.zeros((dev_max_num  ,dev_max_q_num  ),dtype=np.uint32 )
dev_Q_pos   = np.zeros((dev_max_num  ,dev_max_q_num  ),dtype=np.uint32 )
dev_Q_tag   = np.zeros((dev_max_num  ,dev_max_q_num  ),dtype=np.uint32 )
dev_Q_ner   = np.zeros((dev_max_num  ,dev_max_q_num  ),dtype=np.uint32 )
dev_A       = np.zeros((dev_max_num  ,2              ),dtype=np.float32)

for i in range(train_max_num):
    temp_p = list_context[i]
    temp_p = [w2i[word] for word in temp_p]
    temp_p = np.array(Vocab.create_padded_list(temp_p,train_max_p_num),dtype=np.uint32)
    temp_pos = np.array(Vocab.create_padded_list_with_pad_value(list_context_pos[i],train_max_p_num,pos_list['']       ),dtype=np.uint32)
    temp_tag = np.array(Vocab.create_padded_list_with_pad_value(list_context_tag[i],train_max_p_num,pos_tag_list['NIL']),dtype=np.uint32) 
    temp_ner = np.array(Vocab.create_padded_list_with_pad_value(list_context_ner[i],train_max_p_num,ner_list['']       ),dtype=np.uint32)
    train_P[i,:] = temp_p
    train_P_pos[i,:] = temp_pos
    train_P_tag[i,:] = temp_tag
    train_P_ner[i,:] = temp_ner
    temp_q = list_question[i]
    temp_q = [w2i[word] for word in temp_q]
    temp_q = np.array(Vocab.create_padded_list(temp_q,train_max_q_num),dtype=np.uint32)
    temp_pos = np.array(Vocab.create_padded_list_with_pad_value(list_question_pos[i],train_max_q_num,pos_list['']       ),dtype=np.uint32)
    temp_tag = np.array(Vocab.create_padded_list_with_pad_value(list_question_tag[i],train_max_q_num,pos_tag_list['NIL']),dtype=np.uint32) 
    temp_ner = np.array(Vocab.create_padded_list_with_pad_value(list_question_ner[i],train_max_q_num,ner_list['']       ),dtype=np.uint32)
    train_Q[i,:] = temp_q
    train_Q_pos[i,:] = temp_pos
    train_Q_tag[i,:] = temp_tag
    train_Q_ner[i,:] = temp_ner
    train_A[i,:] = spans[i]
for i in range(dev_max_num):
    temp_p = dev_list_context[i]
    temp_p = [w2i[word] for word in temp_p]
    temp_p = np.array(Vocab.create_padded_list(temp_p,dev_max_p_num),dtype=np.uint32)
    temp_pos = np.array(Vocab.create_padded_list_with_pad_value(dev_list_context_pos[i],dev_max_p_num,pos_list['']       ),dtype=np.uint32)
    temp_tag = np.array(Vocab.create_padded_list_with_pad_value(dev_list_context_tag[i],dev_max_p_num,pos_tag_list['NIL']),dtype=np.uint32) 
    temp_ner = np.array(Vocab.create_padded_list_with_pad_value(dev_list_context_ner[i],dev_max_p_num,ner_list['']       ),dtype=np.uint32)
    dev_P[i,:] = temp_p
    dev_P_pos[i,:] = temp_pos
    dev_P_tag[i,:] = temp_tag
    dev_P_ner[i,:] = temp_ner
    temp_q = dev_list_question[i]
    temp_q = [w2i[word] for word in temp_q]
    temp_q = np.array(Vocab.create_padded_list(temp_q,dev_max_q_num),dtype=np.uint32)
    temp_pos = np.array(Vocab.create_padded_list_with_pad_value(dev_list_question_pos[i],dev_max_q_num,pos_list['']       ),dtype=np.uint32)
    temp_tag = np.array(Vocab.create_padded_list_with_pad_value(dev_list_question_tag[i],dev_max_q_num,pos_tag_list['NIL']),dtype=np.uint32) 
    temp_ner = np.array(Vocab.create_padded_list_with_pad_value(dev_list_question_ner[i],dev_max_q_num,ner_list['']       ),dtype=np.uint32)
    dev_Q[i,:] = temp_q
    dev_Q_pos[i,:] = temp_pos
    dev_Q_tag[i,:] = temp_tag
    dev_Q_ner[i,:] = temp_ner
    dev_A[i,:] = dev_spans[i]

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
    pickle.dump(id_to_qid, f, pickle.HIGHEST_PROTOCOL)

with open( './SQUAD/dev/dev_id_to_qid' + '.pkl', 'wb') as f:
    pickle.dump(dev_id_to_qid, f, pickle.HIGHEST_PROTOCOL)

print('preprocess SQUAD data done!')

print('preprocess GloVe data!')
glove_length = get_line_count(glove_dir)
glove_embedding = np.zeros((glove_length, 300), dtype=np.float32)
glove_vocab,glove_embdeeing = load_glove_vocab(glove_dir,glove_embedding)
squad_word_embedding = np.zeros((len(Vocab.w_to_i),300),dtype=np.float32)
for i in range(len(Vocab.w_to_i)-2):
    now_word = Vocab.i_to_w[i]
    if now_word in glove_vocab:
        glove_index = glove_vocab[now_word]
        squad_word_embedding[i,:] = glove_embedding[glove_index,:]
np.save('./SQUAD/squad_word_embedding.npy',squad_word_embedding)
np.save('./Glove/glove_word_embedding.npy',glove_embedding)

print('preprocess GloVe data done!')