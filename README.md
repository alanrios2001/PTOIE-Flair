# port_noie2
A pt-br OpenIE model.

## Setup Project:
1. First install python 3.8 and run pip install poetry.
2. Open the project, set a poetry python interpreter and run poetry update on venv terminal.
3. After that, you need to wait for poetry to install all the needed packages.

## Download pré-trained models:
For downloading my pré-trained models and make it working, i'm leaving a link, just download and
move the folders to 'train_output' folder on the repo clone.
link: https://drive.google.com/drive/folders/1w_yTuIrfLOtluQogalxRTaRzef9dy2m3?usp=share_link



## Train and Pred Usage:

### Training the model
For training the model, first get your data ready, with train, dev and test splits.
run the following command(you can replace with your own parameters):

```cmd
python3 train.py (max_epochs) (model_name) (train_file) (test_file) (dev_file)
```

example1:

```cmd
python3 train.py 150 PTOIE datasets/saida_match PTOIE_train.txt PTOIE_test.txt PTOIE_dev.txt
```

### Making predictions

For predicting, at first, you need a trained model, so if you didnt trained any model, back on the training step.
If you have your trained model, just run:

```cmd
Python3 predict.py (model_name) (sentence)
```
example2:
```cmd
Python3 predict.py PTOIE "A Matemática é uma ciência que utiliza conceitos e técnicas para a formação de conhecimentos abstratos e concretos."
```

## Working example
```cmd
|  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  |
Extração:  Os cachorros são os melhores amigos do homem
Extração:  Os cachorros são mamiferos
|  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  |


|  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  |
|  ------------------------------------------------------------------------------------------------ MAIS INFO -------------------------------------------------------------------------------------------------  |
|  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  |
|  sentença:                                                                                                                                                                                                     |
|   "Os cachorros , que são mamiferos , são os melhores amigos do homem ."                                                                                                                                       |
|  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  |
|  extrações:                                                                                                                                                                                                    |
|   ["Os cachorros"/ARG0, "são"/V, "mamiferos"/ARG1, "são"/V, "os melhores amigos do homem"/ARG1]                                                                                                                |
|  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  |
|  probs:                                                                                                                                                                                                        |
|  [Span[0:2]: "Os cachorros" → ARG0 (0.8672), Span[4:5]: "são" → V (0.6802), Span[5:6]: "mamiferos" → ARG1 (0.3477), Span[7:8]: "são" → V (0.9333), Span[8:13]: "os melhores amigos do homem" → ARG1 (0.5932)]   |
|  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  |
```

## Evaluation

You can run the eval script with the trained model:
```cmd
python3 eval.py (model_dir) (output_txt_name) (corpus_dir) (train_path) (test_path) (dev_path)
```

example5:
```cmd
python3 eval.py train_output/PTOIE PTOIE_eval datasets/saida_match PTOIE_train.txt PTOIE_test.txt PTOIE_dev.txt
```
output:
```cmd
Results:
- F-score (micro) 0.6129
- F-score (macro) 0.6132
- Accuracy 0.4425

By class:
              precision    recall  f1-score   support

        ARG1     0.4731    0.5163    0.4938       153
           V     0.7453    0.7895    0.7668       152
        ARG0     0.5972    0.5621    0.5791       153

   micro avg     0.6038    0.6223    0.6129       458
   macro avg     0.6052    0.6226    0.6132       458
weighted avg     0.6049    0.6223    0.6129       458
```

## Dataset tools

I made available some tools I wrote to convert and create a conll format dataset, feel free to use, some of them is not user friendly to use,
but you can run the main.py to prepare a conll with the labels ready to train with:
- The following command only works with the PTOIE dataset, if you want to use your own dataset, see explanation of example4.
```cmd
python3 datasets/main.py (output_name) (json_dir) (input_dir) (test_size) (dev_size)
```
example3:
```cmd
PTOIE saida_match/json_dump.json PTOIE/PTOIE.txt 0.1 0.1
```

If you want to create a conll of your own dataset, you need a json on the following format example:

```json
{"0": 
  {"Id": 0,
   "sent": " A universidade \u00e9 a sede principal da Congrega\u00e7\u00e3o da Santa Cruz (embora n\u00e3o seja sua sede oficial, que fica em Roma).",
   "ext": [{"arg1": "A sede da congrega\u00e7\u00e3o da santa cruz  ", "rel": "   fica  ", "arg2": "  em Roma"}]},
"1": 
  {"Id": 1,
   "sent": " A universidade \u00e9 afiliada \u00e0 Congrega\u00e7\u00e3o da Santa Cruz (em latim Congregatio a Sancta Cruce, p\u00f3s-nominais abreviados \"CSC\").",
   "ext": [{"arg1": "A congrega\u00e7\u00e3o da santa cruz em latim  ", "rel": "   \u00e9  ", "arg2": "  Congregatio a Sancta Cruce"}]},
```

With the json file, run the following command

```cmd
python3 datasets/main.py (output_name) (test_size) (dev_size) (json_dir) ""
```

example4:
```cmd
python3 datasets/main.py PTOIE 0.1 0.1 saida_match/json_dump.json ""
```
