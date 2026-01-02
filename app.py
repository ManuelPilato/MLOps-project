import os
import time
import sys
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from transformers import pipeline
from prometheus_client import Counter, Histogram, make_asgi_app
import uvicorn

#this part is for grafana using prometheus functions
PREDICTION_COUNTER = Counter('model_predictions_total', 'total number of predictions', ['sentiment'])
LATENCY_HISTOGRAM = Histogram('model_prediction_latency_seconds', 'response time of the model')

#using fastapi we can create an interface with different endpoint (as many as you want)
app = FastAPI(
    title="MLOps Sentiment Analysis API",
    version="1.0.0"
)

#taking the previously saved model inside the right folder
MODEL_PATH = "./sentiment_model"
if not os.path.exists(MODEL_PATH):
    #if it doesn't exists i'll take the model from the link
    MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"

try:
    sentiment_model = pipeline("sentiment-analysis", model=MODEL_PATH, tokenizer=MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    sentiment_model = None


class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    label: str
    score: float
    latency: float


#ENDPOINT 1: home page with welcome message
@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is Online", "docs": "/docs"}


#ENDPOINT 2: model status 
@app.get("/status")
def get_status():
    #Is the system healthy?
    if sentiment_model is None:
        raise HTTPException(status_code=503, detail="Model is not working")
    
    return {"status": "ready" }


#ENDPOINT 3: predictions
@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    
    #start process
    start_time = time.time()
    
    # inference
    result = sentiment_model(request.text)[0]
    
    latency = time.time() - start_time
    
    # saving time for grafaan
    label = result['label'].lower()
    score = float(result['score'])
    
    PREDICTION_COUNTER.labels(sentiment=label).inc()
    LATENCY_HISTOGRAM.observe(latency)
    
    return SentimentResponse(
        text=request.text,
        label=label,
        score=score,
        latency=latency
    )

#ENDPOINT 4: metrics
# from here, prometheus will take data to analyze 
app.mount("/metrics", make_asgi_app())


def tests():
    #launch application
    print("Launch tests")
    client = TestClient(app)

    #endpoint status
    response = client.get("/status")
    assert response.status_code == 200, "Error: Endpoint /status not available"
    print("[OK] Endpoint /status.")

    #i'll make 1 static testfor prediction
    test_text = "MLOps is great!"
    response = client.post("/predict", json={"text": test_text})
    assert response.status_code == 200, "Error: Endpoint /predict failed"
    data = response.json()
    assert data["text"] == test_text, "Error: wrong prediction"
    assert "label" in data, "Generic error"
    print(f"[OK] Endpoint /predict succeded (Label: {data['label']}).")

if __name__ == "__main__":
    
    #input argument --test will launch test function
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        try:
            tests()
            sys.exit(0)  #success
        except Exception as e:
            print(f"PIPELINE FAILED: {e}")
            sys.exit(1) #failed
    else:
        # server on 7860 port as requested in HF once u create space docker
        uvicorn.run(app, host="0.0.0.0", port=7860)