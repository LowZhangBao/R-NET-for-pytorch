
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

dev_P_dir        	 	= r"./SQUAD/dev/dev_P.npy"
dev_Q_dir        	 	= r"./SQUAD/dev/dev_Q.npy"
dev_P_char_all_dir   	= r"./SQUAD/dev/dev_P_char_all.npy"
dev_Q_char_all_dir   	= r"./SQUAD/dev/dev_Q_char_all.npy"
dev_P_char_simple_dir   = r"./SQUAD/dev/dev_P_char_simple.npy"
dev_Q_char_simple_dir   = r"./SQUAD/dev/dev_Q_char_simple.npy"
dev_A_dir        	 	= r"./SQUAD/dev/dev_A.npy"