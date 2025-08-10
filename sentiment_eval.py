import os, pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from dotenv import load_dotenv
load_dotenv()

ART = os.getenv("ART_DIR", "artifacts")
labels = pd.read_csv("eval/labels_sentiment.csv")
preds = pd.read_parquet(f"{ART}/sentiment.parquet")

def label_of_row(r):
    arr = [r["neg"], r["neu"], r["pos"]]
    idx = int(max(range(3), key=lambda i: arr[i]))
    return ["negative","neutral","positive"][idx]

preds["pred_label"] = preds.apply(label_of_row, axis=1)
df = labels.merge(preds[["doc_name","pred_label"]], on="doc_name", how="left")

acc = accuracy_score(df["label"], df["pred_label"])
cm  = confusion_matrix(df["label"], df["pred_label"], labels=["negative","neutral","positive"])

pd.DataFrame({"metric":["accuracy"],"value":[acc]}).to_csv("artifacts/sentiment_metrics.csv", index=False)
pd.DataFrame(cm, index=["neg","neu","pos"], columns=["neg","neu","pos"]).to_csv("artifacts/sentiment_confusion.csv")
print("Accuracy:", acc)
