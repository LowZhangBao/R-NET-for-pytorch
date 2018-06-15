
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

SQUAD_dir = './SQUAD'
GLOVE_dir = './GLOVE'
SQUAD_v1_dir = './SQUAD/v1.1'
Train_v1_dir = './SQUAD/v1.1/train'
DEV_v1_dir   = './SQUAD/v1.1/dev'
SQUAD_v2_dir = './SQUAD/v2.0'
Train_v2_dir = './SQUAD/v2.0/train'
DEV_v2_dir   = './SQUAD/v2.0/dev'
Model_dir = './Model_save'
TEMP_dir  = './TEMP_DATA'

train_v1_filename = "train-v1.1.json"
dev_v1_filename = "dev-v1.1.json"
train_v2_filename = "train-v2.0.json"
dev_v2_filename = "dev-v2.0.json"

glove_char_filename="glove.840B.300d-char.txt"
glove_zip = "glove.840B.300d.zip"
glove_filename = "glove.840B.300d.txt"

glove_url = "http://nlp.stanford.edu/data/"
glove_char_url = "https://raw.githubusercontent.com/minimaxir/char-embeddings/master/"
train_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
dev_url  = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"


squad_data_train_dir=r'./SQUAD/train-v1.1.json'
squad_data_dev_dir=r'./SQUAD/dev-v1.1.json'
glove_char_dir = r'./GloVe/glove.840B.300d-char.txt'
glove_dir = r'./GloVe/glove.840B.300d.txt'

Glove_Word_Embedding_output_dir  = r"./Glove/glove_word_embedding.npy"
SQUAD_Word_Embedding_output_dir  = r"./SQUAD/squad_word_embedding.npy"
Glove_Char_Embedding_output_dir  = r"./Glove/glove_char_embedding.npy"
SQUAD_Char_all_Embedding_output_dir    = r"./SQUAD/squad_char_all_embedding.npy"
SQUAD_Char_simple_Embedding_output_dir = r"./SQUAD/squad_char_simple_embedding.npy"

use_all_char_vocab=False

word_vocab_w2i_dir = r'./SQUAD/SQUAD_Vocab_w_to_i'
word_vocab_i2w_dir = r'./SQUAD/SQUAD_Vocab_i_to_w'
char_simple_vocab_w2i_dir = r'./SQUAD/SQUAD_char_simple_Vocab_w_to_i'
char_simple_vocab_i2w_dir = r'./SQUAD/SQUAD_char_simple_Vocab_i_to_w'
char_all_vocab_w2i_dir = r'./SQUAD/SQUAD_char_all_Vocab_w_to_i'
char_all_vocab_i2w_dir = r'./SQUAD/SQUAD_char_all_Vocab_i_to_w'

train_P_dir      	 	= r"./SQUAD/train/train_P.npy"
train_Q_dir      	 	= r"./SQUAD/train/train_Q.npy"
train_P_char_all_dir 	= r"./SQUAD/train/train_P_char_all.npy"
train_Q_char_all_dir 	= r"./SQUAD/train/train_Q_char_all.npy"
train_P_char_simple_dir = r"./SQUAD/train/train_P_char_simple.npy"
train_Q_char_simple_dir = r"./SQUAD/train/train_Q_char_simple.npy"
train_A_dir      	 	= r"./SQUAD/train/train_A.npy"

train_Q_id_dir = r"./SQUAD/train/train_Q_id.npy"
train_Q_id_to_qid_dir=r"./SQUAD/train/train_id_to_qid.pkl"

dev_P_dir        	 	= r"./SQUAD/dev/dev_P.npy"
dev_Q_dir        	 	= r"./SQUAD/dev/dev_Q.npy"
dev_P_char_all_dir   	= r"./SQUAD/dev/dev_P_char_all.npy"
dev_Q_char_all_dir   	= r"./SQUAD/dev/dev_Q_char_all.npy"
dev_P_char_simple_dir   = r"./SQUAD/dev/dev_P_char_simple.npy"
dev_Q_char_simple_dir   = r"./SQUAD/dev/dev_Q_char_simple.npy"
dev_A_dir        	 	= r"./SQUAD/dev/dev_A.npy"

dev_Q_id_dir = r"./SQUAD/dev/dev_Q_id.npy"
dev_Q_id_to_qid_dir=r"./SQUAD/dev/dev_id_to_qid.pkl"