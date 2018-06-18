# R-NET in Pytorch
* This repository is reproduce the [R-NET: MACHINE READING  COMPREHENSION WITH SELF-MATCHING NETWORKS](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) in Pytorch.
* This repository is designed for the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset.
* This repository reach F1:65.3099% EM:53.8316% in Epoch 4 batch_size 48 without char Embedding input .
* This is my first repository,Any question can opening issues or [email](s2w81234@gmail.com) me.


## Requirements

Please check your perform and software version.

#### General
	* Python >= 3.5.2
#### Python Packages
	* torch >= 0.3.1.post2
	* Spacy == 2.0.0 (Fix version,Don't upgrade and downgrade)
	* jieba
	* requests
	* zipfile
	* json
	* datetime
## Usage
<Step 1> SQuAD and DRCD dataset preprocessing and Generate Training & Dev Data.
```bash
python3 data_prepro.py
```
#### Parameter Description
	* If your want to change some Parameter , like : max passage length & max char length etc...,you can change setting.py document.

<Step 2> Training R-NET and Setting Parameter.
```bash
python3 train.py --hidden_size 75 --dropout 0.2 --batch_size 16 --char_input 1 --emb_input 1 --encoder_concat 1
```
#### Parameter Description
	* data_version: 1 ==> SQUAD v1.1 Dataset
			2 ==> SQUAD v2.0 Dataset (Not working this repository)
			3 ==> DRCD Dataset 
	* hidden_size : the hidden size of RNNs.
	* dropout : all layer dropout ratio.
	* lr : the learning rate of R-NET.
	* batch_size : the size of batch.
	* Char_input : use char encoder input?
	* emb_input  : Embedding change during training? 1 is Yes, 0 is Fix.
	* epoch : train for N epochs.
	* encoder_concat : R-NET encoder concat 3 layer output.
	* seed : Reproducibility paramter.
<Step 3> Generate Prediction json file.
```bash
python3 predict.py --model_dir './Model_save/v1.1/module_char_input_1_emb_input_1_concat_1_hidden_75_batch_size_16/XXX.cpt'  --batch_size 32 --data_version 1 
```

<Step 4> Get the official Score.
```bash
python3 evaluate-v1.1.py ./SQUAD/v1.1/dev-v1.1.json ./Predict_save/v1.1/XXX.json
```
## Implement Detail
	* Use PackSequence to ignore padding length.
	* This repository is not completely reproduce the R-NET.
	* Use Spacy tokenize for SQUAD Dataset.
	* Use jieba tokenize for DRCD Dataset.

