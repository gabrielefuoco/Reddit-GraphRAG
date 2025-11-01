import logging
from neo4j import GraphDatabase, Driver
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.graph.gds_analyzer import GraphAnalyzer
from src.graph.analyzer.summarizer import IdeologySummarizer
from src.utils.config import load_credentials

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- ANALYSIS PIPELINE ---

async def create_alliance_graph(driver: Driver, confidence_threshold: float = 0.85):
    """Creates AGREES_WITH relationships based on shared stances."""
    logging.info(f"Inizio creazione relazioni 'AGREES_WITH' con soglia di confidenza > {confidence_threshold}...")

    alliance_query = """
    MATCH (u1:User)-[:POSTED]->(p1)-[r1:HAS_STANCE]->(e:PoliticalEntity)
    MATCH (u2:User)-[:POSTED]->(p2)-[r2:HAS_STANCE]->(e)
    WHERE u1 <> u2 AND r1.confidence >= $threshold AND r2.confidence >= $threshold AND r1.stance = r2.stance AND elementId(u1) < elementId(u2)
    WITH u1, u2, count(DISTINCT e) AS weight
    MERGE (u1)-[a:AGREES_WITH]-(u2)
    SET a.weight = toFloat(weight)
    RETURN count(a) AS total_alliances
    """

    with driver.session(database="neo4j") as session:
        result = session.run(alliance_query, threshold=confidence_threshold).single()
        total_alliances = result['total_alliances'] if result else 0
        logging.info(f"Creazione completata. Totale alleanze create o aggiornate: {total_alliances}")
    return total_alliances

async def main():
    """Main function to run the graph analysis pipeline."""
    logging.info("=============================================")
    logging.info("AVVIO PIPELINE DI ANALISI DEL GRAFO")
    logging.info("=============================================")

    driver = None
    graph_name = "allianceGraph" # NOme grafo gds@@@@@@@@@@@

    try:
        creds = load_credentials()
        uri, user, password = creds.get("neo4j_uri"), creds.get("neo4j_user"), creds.get("neo4j_password")

        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logging.info("Connessione a Neo4j verificata.")

        # 1. Crea il grafo delle alleanze
        await create_alliance_graph(driver)

        # 2. Esegui la rilevazione delle comunità GDS
        logging.info("Inizio community detection con GDS (Leiden)...")
        analyzer = GraphAnalyzer(driver=driver)

        analyzer.drop_graph(graph_name)

        # Proietta il grafo delle alleanze e esegui Leiden  @@@@
        analyzer.project_alliance_graph(graph_name=graph_name)
        
        # Uso un valore di gamma calibrato per una migliore risoluzione.@@@@@@@@@@@ TESTARE VALORI PIù ALTI
        analyzer.run_leiden(graph_name=graph_name, community_property="communityId", gamma_value=1.4)
        
        logging.info("Community detection completata. I nodi User sono stati aggiornati.")

        # --- FASE 3: NUOVA LOGICA DI RIASSUNTO ---
        logging.info("Avvio riassunto delle ideologie...")
        ideology_summarizer = IdeologySummarizer(driver=driver)
        await ideology_summarizer.summarize_ideologies()
        logging.info("Riassunto delle ideologie completato.")
        # --- FINE NUOVA LOGICA ---

    except Exception as e:
        logging.error(f"Errore critico durante la pipeline di analisi: {e}", exc_info=True)
    finally:
        if driver:
            # Cleanup finale per sicurezza
            try:
                if 'analyzer' in locals():
                    analyzer.drop_graph(graph_name)
            except Exception as cleanup_e:
                logging.error(f"Errore durante il cleanup finale del grafo GDS: {cleanup_e}")
            driver.close()
            logging.info("Connessione a Neo4j chiusa.")

if __name__ == "__main__":
    asyncio.run(main())