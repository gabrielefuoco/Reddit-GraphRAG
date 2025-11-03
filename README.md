
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
