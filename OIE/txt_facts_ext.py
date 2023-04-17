from predict import Predictor
from OIE.datasets.validated_splits.contractions import transform_portuguese_contractions

model = "test_bert"
oie = Predictor(model)
show_triple = True

with open("texto.txt", "r", encoding="utf-8") as f:
    lines = f.read()
    lines = lines.replace("\n", "")
    lines = lines.split(".")[:-1]

exts = []
for line in lines:
    ext = oie.pred(transform_portuguese_contractions(line), False)
    for e in ext:
        ex = []
        for i in e:
            ex.append(i)
        exts.append(ex)

for ex in exts:
    lenght = len(ex)
    extraction = "extração:"
    if lenght > 2:
        for i in range(lenght):
            extraction += f" {ex[i][0]}"
        if show_triple:
            print(extraction + " → " + f"(ARG0: {ex[0][0]})" + f"(REL: {ex[1][0]})" + f"(ARG1: {ex[2][0]})")
        else:
            print(extraction)
