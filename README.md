
# Analizzatore di Opinioni Politiche su Reddit (GraphRAG)

## Panoramica del Progetto

Questo progetto è un sistema end-to-end progettato per analizzare le discussioni politiche sulla piattaforma Reddit. Implementa una sofisticata pipeline **GraphRAG (Retrieval-Augmented Generation)** che trasforma il contenuto testuale non strutturato in un knowledge graph strutturato e interrogabile, ospitato su Neo4j.

L'obiettivo principale è superare i limiti delle tradizionali ricerche basate su keyword e dei sistemi RAG puramente vettoriali, offrendo un'analisi semantica più profonda e contestualizzata delle opinioni politiche. L'interfaccia utente, costruita con Streamlit, fornisce un punto di accesso intuitivo per esplorare i dati e interagire con un agente di ragionamento basato su LangChain.

### Caratteristiche Chiave

* **Recupero Ibrido a Due Stadi**: Il sistema impiega una strategia di recupero gerarchico. Le query vengono prima risolte interrogando la struttura del grafo per identificare entità e relazioni pertinenti (es. chi ha menzionato una figura politica). I risultati vengono poi riordinati semanticamente attraverso la similarità vettoriale per massimizzare la rilevanza.
* **Fallback Vettoriale Dinamico**: Nel caso in cui nessuna entità strutturata venga trovata, il sistema esegue in modo trasparente una ricerca vettoriale pura sull'intero dataset, garantendo che venga sempre fornita una risposta.
* **Arricchimento NLP Avanzato**: I dati raccolti da Reddit vengono sottoposti a un'analisi NLP approfondita, che include:
    * **Estrazione di Entità Nominate (NER)** per identificare figure, organizzazioni e concetti politici.
    * **Rilevamento della Stance** per classificare le opinioni come favorevoli, contrarie o neutrali.
    * **Generazione di Embedding Vettoriali** per catturare il significato semantico del testo.
* **Community Detection e Alleanze Ideologiche**: Vengono applicati algoritmi di analisi di grafi (Leiden) per identificare cluster di utenti con allineamenti ideologici simili, basati sulla coerenza delle loro posizioni espresse.
* **Riassunti Semantici**: Il sistema genera riassunti di alto livello per le ideologie identificate, aggregando e sintetizzando i post e i commenti più rappresentativi di una specifica posizione su un'entità.

## Architettura di Sistema

Il flusso operativo è suddiviso in due fasi principali: una pipeline offline per la costruzione e l'arricchimento del grafo e una pipeline online per l'interrogazione in tempo reale.
<img width="2329" height="2651" alt="Untitled diagram-2025-11-01-114749" src="https://github.com/user-attachments/assets/ba5a9263-2ae8-42b1-8c2c-2b96a0b1577a" />


## Stack Tecnologico

- **Backend e Orchestrazione**: Python 3.10+
    
- **Interfaccia Web**: Streamlit
    
- **Database a Grafo**: Neo4j 5.x
    
- **Ingestione Dati**: PRAW / asyncpraw
    
- **Elaborazione NLP**: spaCy, Hugging Face Transformers
    
- **Embedding Vettoriali**: Sentence-Transformers
    
- **Orchestrazione LLM**: LangChain
    

## Guida all'Installazione

### Prerequisiti

- Python >= 3.10
    
- Un'istanza di Neo4j 5.x (Community o Enterprise) attiva e accessibile.
    
- Credenziali valide per l'API di Reddit.
    
- Un servizio LLM locale compatibile (es. Ollama).
    

### Procedura di Setup

1. **Clonare il repository:**
    
    Bash
    
    ```
    git clone <URL_DEL_REPOSITORY>
    cd <NOME_CARTELLA_PROGETTO>
    ```
    
2. **Creare e attivare un ambiente virtuale:**
    
    Bash
    
    ```
    python3 -m venv venv
    source venv/bin/activate  # Su Windows: .\venv\Scripts\activate
    ```
    
3. **Installare le dipendenze:**
    
    Bash
    
    ```
    pip install -r requirements.txt
    ```
    
4. **Scaricare il modello linguistico per spaCy:**
    
    Bash
    
    ```
    python -m spacy download en_core_web_sm
    ```
    
5. Configurare le variabili d'ambiente:
    
    Creare un file .env nella directory principale del progetto e popolarlo con le proprie credenziali. Questo file non deve essere tracciato da Git.
    
    Snippet di codice
    
    ```
    # Credenziali API Reddit
    REDDIT_CLIENT_ID="IL_TUO_CLIENT_ID"
    REDDIT_CLIENT_SECRET="IL_TUO_CLIENT_SECRET"
    REDDIT_USER_AGENT="NomeUnivocoDelTuoUserAgent v1.0"
    
    # Credenziali Neo4j
    NEO4J_URI="bolt://localhost:7687"
    NEO4J_USER="neo4j"
    NEO4J_PASSWORD="LA_TUA_PASSWORD"
    ```
    

## Flusso Operativo

L'esecuzione completa del sistema è gestita dallo script orchestratore `run_pipeline.py`.

### Esecuzione della Pipeline Completa

Per eseguire l'intera pipeline di ETL (Extract, Transform, Load) e analisi, utilizzare lo script `run_pipeline.py`. Questo script esegue sequenzialmente tutte le fasi necessarie per costruire e analizzare il grafo.

Bash

```
python run_pipeline.py
```

Lo script si occuperà di:

1. **Ingestione e Costruzione**: Eseguire `src.graph.builder` per scaricare i dati da Reddit, processarli e caricarli in Neo4j.
    
2. **Defragmentazione**: Eseguire `src.scripts.defragment_entities` per identificare le entità duplicate.
    
3. **Unione**: Eseguire `src.scripts.merge_entities` per consolidare le entità duplicate nel grafo.
    
4. **Analisi del Grafo**: Eseguire `src.scripts.run_analysis` per la community detection e la generazione dei riassunti ideologici.
    

### Avvio dell'Applicazione Web

Una volta che la pipeline di costruzione del grafo è stata completata, è possibile avviare l'interfaccia utente interattiva.

Bash

```
streamlit run app.py
```

Questo comando avvierà un server locale e aprirà l'applicazione nel browser, consentendo di interrogare il knowledge graph.

## Struttura del Progetto

```
.
├── app.py              # Entry point dell'applicazione Streamlit
├── data/               # Dati grezzi e processati (esclusi da Git)
├── requirements.txt    # Dipendenze Python
├── run_pipeline.py     # Script orchestratore per l'intera pipeline
├── src/                # Codice sorgente del progetto
│   ├── agent/          # Logica per l'agente ReAct
│   ├── data_processing/# Script per ingestione e pulizia dati
│   ├── graph/          # Moduli per la costruzione, analisi e schemi del grafo
│   ├── llm/            # Core per l'interazione con i modelli linguistici
│   ├── nlp/            # Funzioni per analisi NLP (NER, stance, embedding)
│   ├── pipeline/       # Logica della catena RAG e template di query
│   ├── scripts/        # Script ausiliari (merge, defragment, etc.)
│   └── utils/          # Funzioni di utilità (es. gestione configurazione)
```
