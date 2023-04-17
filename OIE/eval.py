import flair
import pathlib
from flair.datasets import DataLoader, ColumnCorpus
from flair.models import SequenceTagger
import typer
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

app = typer.Typer()

def get_dev_result(model_name: str):
    model_path = "evaluations\\" + model_name + "/dev.txt"
    with open(model_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip().split(" ") for line in lines]
        y_true = [line[1] for line in lines if len(line) == 3]
        y_pred = [line[2] for line in lines if len(line) == 3]
        return y_true, y_pred

def get_confusion_matrix(y_true, y_pred):
    labels = list(set(y_true))
    df = pd.DataFrame(
        data=confusion_matrix(y_true, y_pred, labels=labels),
        columns=labels,
        index=labels,
    )
    df.index.name = "Y-True"
    df.columns.name = "Y-Pred"
    return df

def get_measures(y_true, df, model_name, log_txt):
    labels = list(set(y_true))
    tps = {}
    fps = {}
    fns = {}
    for label in labels:
        tps[label] = df.loc[label, label]
        fps[label] = df[label].sum() - tps[label]
        fns[label] = df.loc[label].sum() - tps[label]

    #
    # Global
    #
    micro_averages = {}
    macro_averages = {}

    correct_predictions = sum(tps.values())

    total_predictions = df.values.sum()
    accuracy_global = round(correct_predictions / total_predictions, 4) if total_predictions > 0. else 0.

    log_txt += "#-- Local measures --#\n"
    log_txt += f"True Positives: {tps}\n"
    log_txt += f"False Positives: {fps}\n"
    log_txt += f"False Negatives: {fns}\n"
    log_txt += "\n#-- Global measures --#\n"
    log_txt += f"Correct predictions: {correct_predictions}\n"
    log_txt += f"Total predictions: {total_predictions}\n"
    log_txt += f"Accuracy: {accuracy_global}\n"
    log_txt += "\n"
    print("#-- Local measures --#")
    print("True Positives:", tps)
    print("False Positives:", fps)
    print("False Negatives:", fns)

    print("\n#-- Global measures --#")
    print("Correct predictions:", correct_predictions)
    print("Total predictions:", total_predictions)
    print("Accuracy:", accuracy_global)

    return log_txt


class Eval:
    def __init__(self,
                 model: SequenceTagger,
                 out_txt: str,
                 corpus: flair.data.Corpus
                 ):
        self.corpus = corpus
        self.oie = model
        self.log_txt = ""

        path = pathlib.Path(f"evaluations/{out_txt}")
        path.mkdir(parents=True, exist_ok=True)

        result = self.oie.evaluate(self.corpus.dev,
                                   mini_batch_size=16,
                                   out_path="evaluations/"+out_txt+"/dev"+".txt",
                                   gold_label_type="label",)

        js = result.classification_report

        with open(f"evaluations/{out_txt}/result"+".json", "a") as f:
            json.dump(js, f, indent=4)

        self.log_txt+="------------------------\n"
        self.log_txt+="-- Evaluation results --\n"
        self.log_txt+=result.detailed_results+"\n"
        print("------------------------")
        print("-- Evaluation results --")
        print(result.detailed_results)

@app.command()
def run(model_path: str, corpus_dir: str, train: str, test: str, dev: str):

    out_txt = model_path.split("/")[-1]
    corpus = ColumnCorpus(data_folder=corpus_dir,
                              column_format={0: 'text', 8: "label", 9: "pos", 10: "dep"},
                              train_file=train,
                              test_file=test,
                              dev_file=dev)


    try:
            #carregando melhor modelo
        model = SequenceTagger.load(model_path + "/best-model.pt")
        log_txt = Eval(model=model, out_txt=out_txt, corpus=corpus).log_txt
    except:
        print("best-model.pt not found, trying final-model.pt")
        try:
                #carregando modelo final
            model = SequenceTagger.load(model_path + "/final-model.pt")
            log_txt = Eval(model=model, out_txt=out_txt, corpus=corpus).log_txt
        except:
            print("final-model.pt not found, are you sure you have a model in this folder?")

    model_name = model_path.split("/")[-1]
    y_true, y_pred = get_dev_result(model_name)
    df = get_confusion_matrix(y_true, y_pred)
    log_txt+=str(df)+"\n\n"
    log_txt = get_measures(y_true, df, out_txt, log_txt)
    log_txt+="By label:\n"
    log_txt+=classification_report(y_true, y_pred, digits=4)+"\n"
    with open(f"evaluations/{out_txt}/log"+".txt", "a") as f:
        f.write(log_txt)
    print("\nBy label:")
    print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    app()
