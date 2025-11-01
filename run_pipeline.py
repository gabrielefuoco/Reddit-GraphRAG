import logging
import subprocess
import sys

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ORCHESTRATOR - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Definizione dei comandi della pipeline 
PIPELINE_STAGES = [
    {
        "name": "FASE 1: INGESTIONE (Builder)",
        "command": [sys.executable, "-m", "src.graph.builder"],
        "description": "Caricamento dei dati grezzi da Reddit al grafo (ELT - Extract & Load)."
    },
    {
        "name": "FASE 2: ANALISI (Defragmenter)",
        "command": [sys.executable, "-m", "src.scripts.defragment_entities"],
        "description": "Analisi delle entità nel grafo per creare la mappa di consolidamento."
    },
    {
        "name": "FASE 3: TRASFORMAZIONE (Merger)",
        "command": [sys.executable, "-m", "src.scripts.merge_entities"],
        "description": "Applicazione della mappa per unire i nodi duplicati."
    },
    {
        "name": "FASE 4: ANALISI DEL GRAFO (Community Detection & Summarization)",
        "command": [sys.executable, "-m", "src.scripts.run_analysis"],
        "description": "Esecuzione di algoritmi GDS per community detection e riassunto semantico."
    }
]

def run_stage(stage: dict) -> bool:
    """Esegue un singolo stage mostrando l'output in tempo reale."""
    name = stage["name"]
    command = stage["command"]
    
    logging.info(f"--- AVVIO: {name} ---")
    logging.info(f"Descrizione: {stage['description']}")
    logging.info(f"Comando Eseguito: {' '.join(command)}")

    # Popen per avere il controllo sullo streaming dell'output
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='ignore',  # Ignora i caratteri corrotti.
        bufsize=1
    )

    # stampA l'output riga per riga in tempo reale
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            logging.info(f"[{name}] {line.strip()}")

    return_code = process.wait()

    if return_code == 0:
        logging.info(f"--- SUCCESSO: {name} completato con codice {return_code}. ---")
        return True
    else:
        logging.error(f"--- FALLIMENTO: {name} terminato con codice di errore {return_code}. La pipeline è stata interrotta. ---")
        return False

def main():
    """Orchestra l'esecuzione sequenziale di tutti gli stage."""
    logging.info("======================================================")
    logging.info("AVVIO PIPELINE ELT COMPLETA (Modalità Sincrona Robusta)")
    logging.info("======================================================")

    for stage in PIPELINE_STAGES:
        success = run_stage(stage)
        if not success:
            logging.error("Interruzione della pipeline a causa di un errore nello stage precedente.")
            break
    
    logging.info("======================================================")
    logging.info("PIPELINE ELT TERMINATA")
    logging.info("======================================================")

if __name__ == "__main__":
    main()