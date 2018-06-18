import os

word_embedding_init=None
char_embedding_init=None
word_dim = 300
char_dim = 300

train_p_max=400
train_q_max=50
c_max=16
dev_p_max=None
dev_q_max=None

#p_max = None or specific number
#q_max = None or specific number
#c_max = None or specific number
version=[1,2]

DRCD_word_init='random'
DRCD_char_init='random'

DRCD_dir = './DRCD'
DRCD_train_dir='./DRCD/train'
DRCD_dev_dir='./DRCD/dev'
DRCD_model_dir='./Model_save/DRCD'
DRCD_Prediction_dir = './Predict_save/DRCD'
DRCD_train_filename='DRCD_training.json'
DRCD_dev_filename='DRCD_dev.json'
DRCD_url='https://raw.githubusercontent.com/DRCSolutionService/DRCD/master/'
DRCD_Word_Embedding_file  = 'DRCD_word_embedding.npy'
DRCD_Char_Embedding_file  = 'DRCD_char_embedding.npy'





SQUAD_dir = './SQUAD'
GLOVE_dir = './GLOVE'
FAST_dir  = './FAST'
SQUAD_v1_dir = './SQUAD/v1.1'
Train_v1_dir = './SQUAD/v1.1/train'
DEV_v1_dir   = './SQUAD/v1.1/dev'
SQUAD_v2_dir = './SQUAD/v2.0'
Train_v2_dir = './SQUAD/v2.0/train'
DEV_v2_dir   = './SQUAD/v2.0/dev'
Prediction_dir='./Predict_save/'
Prediction_v1_dir = './Predict_save/v1.1'
Prediction_v2_dir = './Predict_save/v2.0'
Model_dir = './Model_save'
Model_v1_dir = './Model_save/v1.1'
Model_v2_dir = './Model_save/v2.0'
TEMP_dir  = './TEMP_DATA'

train_v1_filename = "train-v1.1.json"
dev_v1_filename = "dev-v1.1.json"
train_v2_filename = "train-v2.0.json"
dev_v2_filename = "dev-v2.0.json"

glove_char_filename="glove.840B.300d-char.txt"
glove_zip = "glove.840B.300d.zip"
glove_filename = "glove.840B.300d.txt"

fast_zh_filename = "wiki.zh.vec"
fast_url = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/"

glove_url = "http://nlp.stanford.edu/data/"
glove_char_url = "https://raw.githubusercontent.com/minimaxir/char-embeddings/master/"
train_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
dev_url  = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"


squad_data_train_v1_dir=r'./SQUAD/'
squad_data_dev_v1_dir=r'./SQUAD/dev-v1.1.json'
squad_data_train_v2_dir=r'./SQUAD/train-v1.1.json'
squad_data_dev_v2_dir=r'./SQUAD/dev-v1.1.json'

glove_char_dir = r'./GloVe/glove.840B.300d-char.txt'
glove_dir = r'./GloVe/glove.840B.300d.txt'

Glove_Word_Embedding_output_dir  = r"./Glove/glove_word_embedding.npy"
SQUAD_Word_Embedding_output_dir  = r"./SQUAD/squad_word_embedding.npy"
Glove_Char_Embedding_output_dir  = r"./Glove/glove_char_embedding.npy"
SQUAD_Char_all_Embedding_output_dir    = r"./SQUAD/squad_char_all_embedding.npy"
SQUAD_Char_simple_Embedding_output_dir = r"./SQUAD/squad_char_simple_embedding.npy"
Fasttext_Word_Embedding_output_dir = r"./FAST/fast_word_embedding.npy"
Fasttext_Char_Embedding_output_dir = r"./FAST/fast_char_embedding.npy"


use_all_char_vocab=False

word_vocab_w2i_file = 'SQUAD_Vocab_w_to_i'
word_vocab_i2w_file = 'SQUAD_Vocab_i_to_w'
char_simple_vocab_w2i_file = 'SQUAD_char_simple_Vocab_w_to_i'
char_simple_vocab_i2w_file = 'SQUAD_char_simple_Vocab_i_to_w'
char_all_vocab_w2i_file = 'SQUAD_char_all_Vocab_w_to_i'
char_all_vocab_i2w_file = 'SQUAD_char_all_Vocab_i_to_w'

train_P_file      	 	 = 'train_P.npy'
train_P_pos_file      	 = 'train_P_pos.npy'
train_P_tag_file      	 = 'train_P_tag.npy'
train_P_ner_file      	 = 'train_P_ner.npy'
train_Q_file      	 	 = 'train_Q.npy'
train_Q_pos_file      	 = 'train_Q_pos.npy'
train_Q_tag_file      	 = 'train_Q_tag.npy'
train_Q_ner_file      	 = 'train_Q_ner.npy'
train_P_char_all_file 	 = 'train_P_char_all.npy'
train_Q_char_all_file 	 = 'train_Q_char_all.npy'
train_P_char_simple_file = 'train_P_char_simple.npy'
train_Q_char_simple_file = 'train_Q_char_simple.npy'
train_A_file      	 	 = 'train_A.npy'

train_Q_id_file = 'train_Q_id.npy'
train_Q_id_to_qid_file='train_id_to_qid.pkl' 

dev_P_file        	 	= 'dev_P.npy'
dev_P_pos_file        	= 'dev_P_pos.npy'
dev_P_tag_file        	= 'dev_P_tag.npy'
dev_P_ner_file        	= 'dev_P_ner.npy'
dev_Q_file        	 	= 'dev_Q.npy'
dev_Q_pos_file        	= 'dev_Q_pos.npy'
dev_Q_tag_file        	= 'dev_Q_tag.npy'
dev_Q_ner_file        	= 'dev_Q_ner.npy'
dev_P_char_all_file   	= 'dev_P_char_all.npy'
dev_Q_char_all_file   	= 'dev_Q_char_all.npy'
dev_P_char_simple_file  = 'dev_P_char_simple.npy'
dev_Q_char_simple_file  = 'dev_Q_char_simple.npy'
dev_A_file        	 	= 'dev_A.npy'

dev_Q_id_file = 'dev_Q_id.npy'
dev_Q_id_to_qid_file='dev_id_to_qid.pkl' 


SQUAD_Word_Embedding_file  = 'squad_word_embedding.npy'
SQUAD_Char_all_Embedding_file    = 'squad_char_all_embedding.npy'
SQUAD_Char_simple_Embedding_file = 'squad_char_simple_embedding.npy'
















