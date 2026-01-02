#includes for the existing model
import os
import sys
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline

#loading the model from huggingn face
def download():
    print("loading the model from hugging face...\n")
    try:
        #text-classification o sentiment-analysis?
        #link in the requirements of the project
        model = pipeline("sentiment-analysis", "cardiffnlp/twitter-roberta-base-sentiment-latest")
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)

    return model


#saving the model locally in the folder sentiment_model
def save(model):
    output_dir = "./sentiment_model"
    #check if the folder already exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"saving model in {output_dir}")

    try:
        model.save_pretrained(output_dir)
        print("Model saved .")

    except Exception as e:
        print(f"Error during saving: {e}")
        sys.exit(1)

#accuracy tests of the model using another dataset
def test_dataset(model):

    #dict
    LABEL_TO_ID = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    try:
        #taking a dataset that fits the model
        ds = load_dataset("tweet_eval", "sentiment",  split="test")

        subset = ds.shuffle(seed=42).select(range(100))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
            
    #print(type(subset[0]))
    #for i in range(0,10):
    #    print(subset[i])


    #splitting 2 columns of the dataset 
    texts = list(subset["text"])         #tweets
    true_labels = list(subset["label"])  #results
    
    #calculating the predictions with the model only passing the texts 
    predictions = model(texts)
    
    #conversion between words and numbers
    pred_labels = []
    for res in predictions:
        txt_label = res['label'] #'positive'
        pred_id = LABEL_TO_ID.get(txt_label) #translate with dict
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
    #if the accuracy is wrose than 60% than it will launch an error in the github unit tests
    assert accuracy >= ACCURACY_THRESHOLD, f"PIPELINE FAILED: Model accuracy {accuracy:.2f} is below threshold {ACCURACY_THRESHOLD}"
    print("[OK] Quality Check Passed.")

    #saving model locally
    save(model)