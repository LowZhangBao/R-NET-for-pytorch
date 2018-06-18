import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import data
import random
import datetime
import pickle
import numpy as np
import json
import setting
from Vocab import Vocab_class   
from util import load_squad_data,create_mask,get_data_engine


parser = argparse.ArgumentParser()

parser.add_argument('--data_version',type=int,default=1,help='choose squad database vesrion 1 or 2')
args = parser.parse_args()


if __name__ == '__main__':
    if args.data_version==1:
        Model_base = setting.Model_v1_dir
        SQUAD_dir  = setting.SQUAD_v1_dir
        train_dir  = setting.Train_v1_dir
        dev_dir    = setting.DEV_v1_dir
        print('choose SQUAD v1.1 Dataset')
    elif args.data_version==2:
        Model_base = setting.Model_v2_dir
        SQUAD_dir  = setting.SQUAD_v2_dir
        train_dir  = setting.Train_v2_dir
        dev_dir    = setting.DEV_v2_dir
        print('choose SQUAD v2.0 Dataset')

    elif args.data_version==3:
        Model_base = setting.DRCD_model_dir
        SQUAD_dir  = setting.DRCD_dir
        train_dir  = setting.DRCD_train_dir
        dev_dir    = setting.DRCD_dev_dir
        print('choose DRCD Dataset')

        setting.use_all_char_vocab=True
    else:
        Model_base = setting.Model_v1_dir
        SQUAD_dir  = setting.SQUAD_v1_dir
        train_dir  = setting.Train_v1_dir
        dev_dir    = setting.DEV_v1_dir
        print('not this version,Auto choose SQUAD v1.1 Dataset.')
    train_P,train_Q,train_P_c,train_Q_c,train_A,dev_P,dev_Q,dev_P_c,dev_Q_c,dev_A = load_squad_data(version_flag=args.data_version)
 
    word_Vocab = Vocab_class()
    word_Vocab.load(os.path.join(SQUAD_dir,setting.word_vocab_w2i_file),os.path.join(SQUAD_dir,setting.word_vocab_i2w_file))
    word_embedding=np.load(os.path.join(SQUAD_dir,setting.SQUAD_Word_Embedding_file))

    #Get char Vocab and Embedding

    char_Vocab  = Vocab_class()
    if setting.use_all_char_vocab is True:
        char_Vocab.load(os.path.join(SQUAD_dir,setting.char_all_vocab_w2i_file) ,os.path.join(SQUAD_dir,setting.char_all_vocab_i2w_file))
        char_embedding=np.load(os.path.join(SQUAD_dir,setting.SQUAD_Char_all_Embedding_file))
    else:
        char_Vocab.load(os.path.join(SQUAD_dir,setting.char_simple_vocab_w2i_file),os.path.join(SQUAD_dir,setting.char_simple_vocab_i2w_file))
        char_embedding=np.load(os.path.join(SQUAD_dir,setting.SQUAD_Char_simple_Embedding_file))


    word_PAD_ID = word_Vocab.PAD_ID
    word_UNK_ID = word_Vocab.UNK_ID
    char_PAD_ID = char_Vocab.PAD_ID
    char_UNK_ID = char_Vocab.UNK_ID
    with open(os.path.join(train_dir,setting.train_Q_id_to_qid_file),'rb') as f:
        id_to_qid = pickle.load(f)
    
    Q_id = np.load(os.path.join(train_dir,setting.train_Q_id_file))

    with open(os.path.join(dev_dir,setting.dev_Q_id_to_qid_file),'rb') as f:
        dev_id_to_qid = pickle.load(f)
    
    dev_Q_id = np.load(os.path.join(dev_dir,setting.dev_Q_id_file))

    print('train phase')
    print('Word Vocab size: %d | Char Vocab size: %d | Max context: %d | Max question: %d'%(
          word_embedding.shape[0],char_embedding.shape[0], train_P.shape[1], train_Q.shape[1]))
    print('dev phase')
    print('Word Vocab size: %d | Char Vocab size: %d | Max context: %d | Max question: %d'%(
          word_embedding.shape[0],char_embedding.shape[0], dev_P.shape[1], dev_Q.shape[1]))

    prediction_dict={}
    for i in range(dev_P.shape[0]):
        if np.isnan(dev_A[i,0]) or np.isnan(dev_A[i,1]):
            sentense= ""
        else:
            s_index = int(dev_A[i,0])
            e_index = int(dev_A[i,1])
            ans_len = abs(e_index-s_index)+1
            sentense= ""
            for j in range(ans_len):
                if j!=0:
                    sentense+=" "
                if s_index>=e_index:
                    sentense +=  word_Vocab.get_word(dev_P[i,s_index-j])
                else:
                    sentense +=  word_Vocab.get_word(dev_P[i,s_index+j])
        prediction_dict[dev_id_to_qid[dev_Q_id[i]]] = sentense

    


    with open('original_dev_'+str(args.data_version)+'.json','w') as f:
        json.dump(prediction_dict,f)

    prediction_dict={}
    for i in range(train_P.shape[0]):
        if np.isnan(train_A[i,0]) or np.isnan(train_A[i,1]):
            sentense= ""
        else:
            s_index = int(train_A[i,0])
            e_index = int(train_A[i,1])
            ans_len = abs(e_index-s_index)+1
            sentense= ""
            for j in range(ans_len):
                if j!=0:
                    sentense+=" "
                if s_index>=e_index:
                    sentense +=  word_Vocab.get_word(train_P[i,s_index-j])
                else:
                    sentense +=  word_Vocab.get_word(train_P[i,s_index+j])
        prediction_dict[id_to_qid[Q_id[i]]] = sentense

    


    with open('original_train_'+str(args.data_version)+'.json','w') as f:
        json.dump(prediction_dict,f)        
