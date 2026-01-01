#includes for the existing model
import os
import sys
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline

def download():
    print("loading the model from hugging face...\n")
    try:
        #text-classification o sentiment-analysis?
        model = pipeline("sentiment-analysis", "cardiffnlp/twitter-roberta-base-sentiment-latest")
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)

    return model

def save(model):
    output_dir = "./model"
    print(f"saving model in {output_dir}")

    model.save_pretrained(output_dir)
    print("Model saved correctly.")


def test_dataset(model):

    #dict
    LABEL_TO_ID = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    #taking a dataset that fits the model
    ds = load_dataset("tweet_eval", "sentiment",  split="test")

    subset = ds.shuffle(seed=42).select(range(100))
    
    #for i in range(0,10):
    #    print(subset[i])

    texts = list(subset["text"])         #tweets
    true_labels = list(subset["label"])  #results
    
    predictions = model(texts)
    
    #conversion between words and numbers
    pred_labels = []
    for res in predictions:
        txt_label = res['label'] #'positive'
        pred_id = LABEL_TO_ID.get(txt_label) #translate
        pred_labels.append(pred_id)

    # Calculate metrics difference from true ones and predicted
    acc = accuracy_score(true_labels, pred_labels)
    
    print(f"Accuracy: {acc:.2%}")
    print("Detailed Report:")
    print(classification_report(true_labels, pred_labels, target_names=["Negative", "Neutral", "Positive"]))
    
    return acc

if __name__ == "__main__":

    #taking the model from hf
    model = download()

    #unit test with accuracy
    accuracy = test_dataset(model)
    ACCURACY_THRESHOLD = 0.6
    print(f"[*] Quality Check: Is Accuracy ({accuracy:.2f}) > Threshold ({ACCURACY_THRESHOLD})?")
    assert accuracy >= ACCURACY_THRESHOLD, f"PIPELINE FAILED: Model accuracy {accuracy:.2f} is below threshold {ACCURACY_THRESHOLD}"
    print("[OK] Quality Check Passed.")

    #saving model locally
    save(model)