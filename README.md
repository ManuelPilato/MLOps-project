---
title: MLOps Sentiment Analysis
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# MLOps Project: Sentiment Analysis Pipeline

Implementazione completa di una pipeline MLOps per l'analisi del sentiment, progettata per automatizzare l'intero ciclo di vita del machine learning: dall'acquisizione del modello, alla validazione, fino al deployment in ambiente e monitoraggio continuo.

## Panoramica del Progetto

L'obiettivo principale del progetto è stato creare un sistema robusto e scalabile in grado di classificare testi (tweet/commenti) in tre categorie di sentiment: positivo, negativo, neutro. Invece di limitarsi alla creazione del modello, il focus è stato posto sull'ingegnerizzazione del processo (CI/CD) e sull'osservabilità del sistema in produzione.

## Scelte Progettuali e Architettura

Durante lo sviluppo sono state prese decisioni architetturali specifiche per garantire modularità e affidabilità:

1.  **Modello (RoBERTa vs FastText)**: Come da requisiti, è stato usato il modello preaddestrato `cardiffnlp/twitter-roberta-base-sentiment-latest`, in seguito è stato fatto un Fine Tuning su un dataset a piacimento. 

2.  **Serving (FastAPI)**: Per l'interfaccia utente, è stato usato FastAPI, per sviluppare un'architettura a microservizi reale, esponendo endpoint REST standard (`/predict`, `/status`, `/metrics`) essenziali per l'integrazione con sistemi di monitoraggio esterni.

3.  **Containerizzazione (Docker)**: Per garantire che l'applicazione funzionasse in modo identico sia in locale che sui server di Hugging Face, l'intero stack è stato incapsulato in un container Docker. Questo ha eliminato i problemi di compatibilità tra diverse versioni di librerie Python.

## Fasi di Sviluppo e Metodologia

Il progetto è stato strutturato in fasi sequenziali, con un occhio di riguardo alla validazione automatica.

### Fase 1: Implementazion del Modello
È stato sviluppato uno script di training (`train.py`) che non si limita a scaricare il modello, ma esegue una validazione immediata su un dataset di test (`tweet_eval`). È stato introdotto anche un test sulla bontà delle risposte del modello: infatti se l'accuratezza del modello è inferiore a una soglia prestabilita (60%), la pipeline si interrompe preventivamente, impedendo il rilascio di modelli degradati. Questo è stato automatizzato tramite actions di Github (fase CI).

### Fase 2: Testing Intensivo
Questa è stata la fase più critica e lunga del progetto. Sono stati implementati due livelli di test:
* **Test Unitari**: Verifica delle funzioni di base del modello.
* **Test di Integrazione**: All'interno dell'API stessa (`app.py`), è stata integrata una routine di test che simula chiamate HTTP reali prima dell'avvio del server. Questo garantisce che l'immagine Docker, una volta costruita, sia perfettamente funzionante prima di essere esposta al pubblico.

### Fase 3: CI/CD con GitHub Actions
L'automazione è gestita tramite GitHub Actions. Ogni commit sul ramo principale innesca una pipeline che esegue, nell'ordine: installazione dipendenze, validazione modello, test di integrazione API e, solo in caso di successo, il deployment.

### Fase 4: Monitoraggio e Osservabilità
È stato integrato un sistema di monitoraggio basato su Prometheus. L'applicazione espone metriche tecniche (latenza, errori HTTP) e metriche di business (distribuzione del sentiment predetti), permettendo la visualizzazione dei dati su dashboard Grafana per rilevare eventuali drift del modello o problemi di performance.

## Sfide Riscontrate e Risoluzioni

Durante il percorso sono emerse diverse criticità tecniche che hanno richiesto interventi mirati:

* **Deployment su Hugging Face Spaces**: Una delle sfide principali ha riguardato la configurazione del target di upload. Inizialmente, la pipeline caricava i file nel repository del "Modello" invece che nello "Space". Il problema è stato risolto specificando correttamente il flag `--repo-type space` nella CLI di Hugging Face e configurando adeguatamente i permessi del token.
* **Gestione dei Percorsi in Docker**: L'allineamento tra la struttura delle cartelle locale e quella interna al container Docker ha richiesto diverse iterazioni, specialmente per garantire che il modello salvato localmente dalla pipeline venisse copiato correttamente nella directory di lavoro del container.
* **Permessi Utente**: Per conformarsi alle policy di sicurezza di Hugging Face, è stato necessario configurare il Dockerfile per utilizzare un utente non-root (UID 1000), modificando di conseguenza i permessi di scrittura e le variabili d'ambiente.

## Utilizzo dell'API

L'applicazione è attualmente in esecuzione su Hugging Face Spaces.

**Endpoint Principale:** `POST /predict`

Esempio di corpo della richiesta (JSON):
```json
{
  "text": "I love MLOps"
}
```

**Endpoint per controllare lo stato del modello:** `GET /status`

**Endpoint per vedere le metriche del modello:** `GET /metrics`





## Analisi grafica del modello con Grafana:
<img width="1470" height="802" alt="Screenshot 2026-01-02 at 17 58 58" src="https://github.com/user-attachments/assets/2dada290-cc1d-41f8-bc29-fbdfca24c776" />


## Hugging Face
è possibile trovare il modello nella seguente pagina Hugging Face: https://huggingface.co/MPili22
oppure, per provare il modello, usare l'URL diretto: https://mpili22-mlops.hf.space (seguito da /docs per usare l'interfaccia grafica) 
