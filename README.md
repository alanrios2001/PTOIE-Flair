# port_noie2
A pt-br OpenIE model.

## UPDATE
New best pre-trained model(TradOIE_v3) available, just doenload it on the Google Drive link above.


## Setup Project:
1. First install python 3.8 and run pip install poetry.
2. Open the project, set a poetry python interpreter and run poetry update on venv terminal.
3. After that, you need to wait for poetry to install all the needed packages.

## Download pré-trained models:
For downloading my pré-trained models and make it working, i'm leaving a link, just download and
move the folders to 'train_output' folder on the repo clone.
link: https://drive.google.com/drive/folders/1w_yTuIrfLOtluQogalxRTaRzef9dy2m3?usp=share_link
-Some models paths on drive have fine_tuned versions inside, they had improved performance on gold_dataset eval


## Making predictions

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
