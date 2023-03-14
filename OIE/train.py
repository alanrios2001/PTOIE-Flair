import pathlib
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings, OneHotEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
from trainers.trainer import ModelTrainer
from madgrad import MADGRAD
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
import typer

app = typer.Typer()


@app.command()
def train(epochs: int, name: str, folder: str, train:str, test:str, dev:str):
    # define the structure of the .datasets file
    folders = ['conll2bioes_output/', 'saida_match/', "merges/", "lowered_datasets/"]
    corpus = ColumnCorpus(data_folder=folder,
                          column_format={0: 'text', 8: 'label', 9: "pos", 10: "dep", 11: "ner"},
                          train_file=train,
                          test_file=test,
                          dev_file=dev
                          )

    label_type = "label"    # criando dicionario de tags
    label_dictionary = corpus.make_label_dictionary(label_type=label_type)
    print(label_dictionary)

    emb = TransformerWordEmbeddings(
        "neuralmind/bert-base-portuguese-cased",
        #layers="-1",
        #pooling_operation="first-last",
        #fine_tune=True,
    )

    embedding_types = [
        #OneHotEmbeddings.from_corpus(corpus=corpus, field='pos', min_freq=2, embedding_length=600),
        #OneHotEmbeddings.from_corpus(corpus=corpus, field='dep', min_freq=2, embedding_length=600),
        emb,
        FlairEmbeddings('pt-forward'),
        FlairEmbeddings('pt-backward')
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    # inicializando sequence tagger
    oie = SequenceTagger(hidden_size=2048,
                         embeddings=embeddings,
                         tag_dictionary=label_dictionary,
                         tag_type=label_type,
                         rnn_layers=2,
                         )

    pathlib.Path(f"train_output").mkdir(parents=True, exist_ok=True)

    # inicializando trainer
    trainer = ModelTrainer(oie, corpus)
    # iniciando treino
    trainer.train(f"train_output/{name}",
                  learning_rate=0.005,
                  min_learning_rate=0.0001,
                  mini_batch_size=16,
                  max_epochs=epochs,
                  patience=2,
                  embeddings_storage_mode='cpu',
                  optimizer=MADGRAD,
                  use_amp=True,
                  save_final_model=False,
                  anneal_factor=0.25
                  )


if __name__ == "__main__":
    app()
