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
	* requests
	* zipfile
	* json
	* datetime
## Usage
<Step 1> SQuAD dataset preprocessing and Generate Training & Dev Data.
```bash
python3 squad_data_prepro.py
```

<Step 2> Training R-NET and Setting Parameter.
```bash
python3 train.py --hidden_size 75 --dropout 0.2 --batch_size 16 
```
#### Parameter Description
	* hidden_size : the hidden size of RNNs.
	* dropout : all layer dropout ratio.
	* lr : the learning rate of R-NET.
	* batch_size : the size of batch.
	* epoch : train for N epochs.
	* encoder_concat : R-NET encoder concat 3 layer output.
	* seed : Reproducibility paramter.
<Step 3> Generate Prediction json file.
```bash
python3 predict.py --model_dir module1.cpt --output_name prediction_answer.json
```
<Step 4> Get the official Score.
```bash
python3 evaluate-v1.1.py ./SQUAD/dev-v1.1.json ./prediction_answer.json
```
## Implement Detail
	* Use PackSequence to ignore padding length.
	* This repository is not completely reproduce the R-NET.
	* No Char-embedding input.
	* Use Spacy tokenize.
## To do list
	* Char-embedding input.
