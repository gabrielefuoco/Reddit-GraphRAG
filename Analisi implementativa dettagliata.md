## Obiettivo

L'obiettivo è creare un'applicazione web con Streamlit per interrogare un dataset politico di Reddit tramite GraphRAG. Questa versione si concentra sulla profondità dell'analisi semantica, sulla robustezza della costruzione del grafo e sull'efficienza della pipeline di interrogazione. La strategia è un **retrieval ibrido a due stadi con fallback dinamico**: il grafo genera candidati, il vettore li riordina. Niente fusioni ingenue, solo un workflow brutale ed efficiente.

## 1. Pipeline GraphRAG con Arricchimento Semantico Avanzato

### 1.1. Ingestione e Preprocessing dei Dati

#### **Tecnologia SOTA**

- Raccolta Dati: PRAW (Python Reddit API Wrapper) rimane lo strumento di elezione. Per un prototipo, è sufficiente eseguire uno script batch per raccogliere un dataset statico (es. i top 1000 post dell'ultimo anno da subreddit selezionati come r/politics, r/conservative, r/liberal).
    
- Preprocessing: spaCy per la sua efficienza e le pipeline pre-addestrate.
    

#### **Dettagli Implementativi e Criticità**

- Gestione Limiti API di Reddit: L'API di Reddit limita la raccolta storica a circa 1000 post per ogni endpoint (/top, /new, ecc.). Per un prototipo, questo è un limite operativo da documentare, non un blocco. Aggirarlo richiederebbe archivi esterni (es. Pushshift.io), fuori scopo per la V1.
    
- Pratiche Etiche: È obbligatorio impostare un user_agent descrittivo e rispettare i rate limit. PRAW gestisce il backoff, ma il monitoraggio è responsabilità dello sviluppatore. Vengono raccolti solo dati da subreddit pubblici, escludendo PII.
    
- Pipeline di Preprocessing:
    
        1. Normalizzazione del Testo: Convertire il testo in minuscolo, rimuovere URL, caratteri speciali e punteggiatura non essenziale.
    
        2. Token Cleaning: Rimuovere le stop word (parole comuni come "the", "is", "in") utilizzando le liste fornite da spaCy.
    
        3. Lemmatizzazione: Ridurre le parole alla loro forma base (es. "running" -> "run") per consolidare il significato. La lemmatizzazione è superiore allo stemming perché produce parole di senso compiuto.
    

### 1.2. Costruzione del Knowledge Graph (Unificato)

#### **Tecnologia SOTA**

- Estrazione di Entità (NER): dslim/bert-base-NER. Un modello Transformer fine-tuned, standard de facto per prototipazione rapida per il suo equilibrio tra performance e risorse.
    
- Qualificazione delle Relazioni: L'approccio si concentra sulla qualificazione di relazioni semplici. Si parte da un arco base come MENTIONS. Successivamente, modelli più specifici (come il classificatore NLI per la stance) o chiamate mirate a un LLM vengono utilizzati per arricchire questo arco, trasformandolo in HAS_STANCE con proprietà dettagliate (stance, confidence).
    
- Database a Grafo con Funzionalità Vettoriali: Neo4j (v5.11+). È la scelta obbligata. Oltre al linguaggio Cypher, al supporto GDS e all'integrazione Python, la sua capacità di gestire indici vettoriali nativi elimina la necessità di store esterni, semplificando drasticamente lo stack architetturale.
    
- Generazione Embedding: sentence-transformers/all-mpnet-base-v2. Un modello SOTA per generare embedding vettoriali densi e semanticamente ricchi dal testo. Questi vettori verranno memorizzati direttamente in Neo4j.
    

#### **Struttura-Tipo del Grafo (Schema Dettagliato)**

Il grafo sarà modellato con i seguenti nodi e archi. **Ogni nodo `Post` e `Community` avrà una proprietà `embedding` (un vettore numerico) che verrà indicizzata e interrogata direttamente da Neo4j tramite il suo indice vettoriale nativo.**

|   |   |   |   |
|---|---|---|---|
|**Tipo Elemento**|**Label**|**Proprietà Chiave**|**Descrizione**|
|**Nodo**|`Post`|`id`, `content`, `author`, `timestamp`, `subreddit`, `score`, `embedding`|Rappresenta un post o un commento di Reddit. Contiene il vettore semantico.|
|**Nodo**|`PoliticalEntity`|`id` (es. "Joe Biden"), `name`, `type` (es. "PERSONA", "PARTITO")|Un'entità politica estratta dal testo.|
|**Nodo**|`User`|`name`|L'autore di un post.|
|**Nodo**|`Community`|`id`, `level`, `summary`, `stance_profile`, `embedding`|Un cluster di discussione identificato. Contiene il vettore semantico del sommario.|
|**Arco**|`POSTED`|`timestamp`|`(User) --> (Post)`|
|**Arco**|`REPLY_TO`|`timestamp`|`(Post) --> (Post)`|
|**Arco**|`MENTIONS`|`sentence` (la frase in cui avviene la menzione)|`(Post) --> (PoliticalEntity)`|
|**Arco**|`HAS_STANCE`|`stance` ("FAVOREVOLE", "CONTRARIO", "NEUTRALE"), `confidence` (float, 0.0-1.0)|**(Post) --> (PoliticalEntity)**. La proprietà `confidence` **NON È OPZIONALE**. Rappresenta il punteggio softmax del classificatore. È l'unica difesa contro il rumore del modello.|
|**Arco**|`PART_OF`||`(Post) --> (Community)`|

### 1.3. Analisi Semantica Approfondita (Stance Detection) - REVISIONE CRITICA

#### **Tecnologia SOTA**

- Stance Detection Zero-Shot: `mlburnham/Political_DEBATE_large_v1.0`. Rimane la scelta per la V1, ma con una consapevolezza critica dei suoi limiti.
    

#### **Dettagli Implementativi e Gestione del Rischio Modello**

L'output del classificatore non è più un'etichetta binaria, ma una **coppia (`stance`, `confidence`)**. Questo è un requisito non negoziabile.

1. **Cattura della Confidenza:** Il risultato della stance detection viene aggiunto come proprietà sull'arco `(Post)-[r:HAS_STANCE]->(PoliticalEntity)`. L'arco `r` avrà due proprietà: `r.stance` (l'etichetta predetta) e `r.confidence` (il punteggio grezzo, es. 0.95).
    
2. **Imperativo di Filtraggio:** Qualsiasi query o analisi successiva (in particolare la costruzione del grafo di alleanze ideologiche) **DEVE** usare la confidenza come filtro. Esempio: si considerano valide solo le relazioni `HAS_STANCE` con `confidence > 0.85`. Ignorare questo passaggio significa costruire un grafo su fondamenta di sabbia, inquinato da sarcasmo e ambiguità interpretati erroneamente dal modello.
    
3. **Roadmap per V2 (Obbligatoria):** Pianificare una fase di fine-tuning del modello su un dataset di esempi problematici (sarcasmo, linguaggio codificato) etichettati internamente. Un modello generico non sarà mai sufficiente per un'analisi politica seria.
    

### 1.4. Community Detection (Revisione Strategica)

#### **Tecnologia SOTA**

- Algoritmo: Leiden. Un miglioramento diretto di Louvain, produce comunità meglio connesse. Implementato in igraph, cdlib o direttamente con la libreria GDS di Neo4j per performance superiori.
    
- Riassunti delle Comunità: LLM open-source efficiente (es. Meta Llama 3 8B Instruct) per generare i sommari.
    

#### **Dettagli Implementativi**

L'approccio è stato ridefinito per identificare **alleanze ideologiche** e generare riassunti semanticamente densi.

1. Definizione del Sottografo di Analisi (Grafo di Alleanza Ideologica):
    
        - La community detection non viene eseguita sul grafo completo. Viene eseguita su un grafo proiettato in memoria o tramite GDS.
    
        - Nodi: User.
    
        - Archi: Si crea un arco (u1)-[:AGREES_WITH]->(u2) se e solo se entrambi gli utenti hanno espresso la stessa stance sulla stessa entità politica.
    
        - Peso dell'Arco: Il peso della relazione AGREES_WITH è il numero di entità distinte su cui i due utenti concordano.
    
        - L'algoritmo di Leiden viene eseguito su questo grafo pesato per identificare cluster di utenti con allineamento ideologico.
    
2. Algoritmo di Riassunto Ibrido (Centroide Semantico):
    
        - Per ogni community (cluster di utenti) identificata:
    
        - Fase 1 (Raccolta e Filtro): Si recuperano tutti i post scritti dagli utenti della community. Si filtrano i Top K post (es. K=100) basandosi su una metrica di popolarità (es. score di Reddit).
    
        - Fase 2 (Calcolo Centroide): Si calcola il vettore medio (centroide) degli embedding dei K post selezionati. Questo vettore rappresenta il cuore semantico della discussione.
    
        - Fase 3 (Estrazione Esemplari): Si identificano i Top N post (es. N=3-5) i cui embedding sono più vicini (similarità coseno) al centroide. Questi sono i post più rappresentativi.
    
        - Fase 4 (Generazione Riassunto): Si passa il testo solo di questi N post all'LLM per generare un riassunto denso e mirato.
    

## 2. Architettura e Pipeline di Interrogazione (Revisione Strategica)

### 2.1. Architettura Prototipo (Logica a Due Stadi)

L'architettura non è più una fusione simultanea, ma una **pipeline sequenziale con branching logico**. Questo garantisce prevedibilità e controllo.

```
flowchart TD
    A[Frontend Streamlit] --> B[Input Utente -> Orchestratore]
    B --> C[1. Estrazione Entità dalla Query]
    C --> D[2. Candidate Generation: Query Strutturale su Grafo Neo4j<br><i>Trova tutti i post che menzionano le entità</i>]
    D -- Candidati Trovati --> E[3a. Re-ranking Semantico<br><i>Ordina i candidati per similarità vettoriale con la query</i>]
    D -- Nessun Candidato --> F[3b. Fallback Dinamico<br><i>Esegui una query vettoriale pura su tutto il DB</i>]
    E --> G[4. Recupero Contesto Unificato]
    F --> G
    G --> H[5. Generazione Risposta LLM]
    H --> I[Output Streamlit con Metadati di Match]
```

### 2.2. Dettagli Implementativi per l'Output

- Tecnologia di Visualizzazione: streamlit-agraph per renderizzare grafi interattivi. In alternativa, Pyvis per generare un HTML embeddabile.
    



## 3. Asset di Progetto

### 3.1. File `requirements.txt`

```
# ==============================================================================
# File di Requisiti per l'Analizzatore di Opinioni Politiche su Reddit
# ==============================================================================
# Questo file elenca tutte le dipendenze Python necessarie per il progetto.
# Per installare tutto, esegui: pip install -r requirements.txt
# ------------------------------------------------------------------------------
# --- Core Frameworks ---
streamlit>=1.30.0
# --- Data Handling & Ingestion ---
praw>=7.7.0
pandas>=2.0.0
requests>=2.30.0
# --- NLP & Machine Learning ---
spacy>=3.7.0
# Nota: python -m spacy download en_core_web_trf
transformers>=4.38.0
torch>=2.0.0
sentence-transformers>=2.2.0  # Per la generazione di embedding di testo
# --- LLM Orchestration & Interaction ---
langchain>=0.1.10
# --- Graph Database & Manipulation ---
neo4j>=5.17.0
# --- Visualization ---
plotly>=5.18.0
streamlit-agraph>=0.0.8
pyvis>=0.3.2
# --- Utilities ---
python-dotenv>=1.0.0
```

### 3.2. Struttura delle Cartelle e dei File

```
reddit_analyzer/
│
├──.env
├──.gitignore
├── README.md
├── requirements.txt
│
├── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_testing.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing/
│   ├── graph/
│   ├── nlp/
│   ├── pipeline/
│   └── utils/
│
└── tests/
    ├── __init__.py
    ├── test_ingestion.py
    └── test_nlp.py
```

### 3.3. Descrizione dei File e delle Cartelle Chiave

- app.py: Punto di ingresso dell'applicazione Streamlit.
    
- data/: Contiene i dati.
    
        - raw/: Dati grezzi da Reddit.
    
        - processed/: Dati puliti e pre-elaborati.
    
- src/: Codice sorgente del progetto.
    
        - data_processing/ingestion.py: Script per scaricare i dati da Reddit.
    
        - graph/builder.py: Script per costruire e popolare il grafo Neo4j.
    
        - graph/community_analyzer.py: Script per la community detection.
    
        - nlp/analysis.py: Funzioni per NER, stance, embedding e riassunti.
    
        - pipeline/rag_chain.py: Logica di orchestrazione della pipeline GraphRAG a due stadi.
    
        - utils/config.py: Gestione delle configurazioni e delle credenziali.
    
- notebooks/: Jupyter notebooks per l'esplorazione e la sperimentazione.
    
- .env: File per le variabili d'ambiente (non tracciato da Git).
    
- README.md: Panoramica del progetto e istruzioni di setup.
    

