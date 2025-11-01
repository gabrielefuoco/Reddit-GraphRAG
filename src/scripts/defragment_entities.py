import json
import logging
from itertools import combinations

import numpy as np
from neo4j import GraphDatabase
from rapidfuzz import fuzz # Assicurati di usare rapidfuzz
from sklearn.cluster import AgglomerativeClustering

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import load_credentials

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- PARAMETRI DI CONTROLLO ---
SIMILARITY_THRESHOLD = 85.0
OUTPUT_FILE = "canonical_map.json"

def fetch_unique_entities(driver):
    """Estrae tutti i nomi unici delle entità dal database."""
    logging.info("Fetching unique entity names from Neo4j...")
    query = "MATCH (e:PoliticalEntity) RETURN DISTINCT e.name AS name"
    with driver.session(database="neo4j") as session:
        results = session.run(query)
        entities = [record["name"] for record in results]
    logging.info(f"Found {len(entities)} unique entity names.")
    return entities

def build_clusters(entities: list[str], threshold: float) -> dict[int, list[str]]:
    """
    Raggruppa le entità simili usando clustering gerarchico basato sulla similarità di stringa.
    """
    if not entities:
        return {}

    logging.info("Calculating similarity matrix...")
    distance_matrix = np.full((len(entities), len(entities)), 100.0)

    for i, j in combinations(range(len(entities)), 2):
        # algo più intelligente che ignora l'ordine delle parole e gestisce le inclusioni
        similarity = fuzz.token_set_ratio(entities[i], entities[j])
        distance = 100.0 - similarity
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

    np.fill_diagonal(distance_matrix, 0)

    logging.info("Performing agglomerative clustering...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=(100.0 - threshold)
    )
    labels = clustering.fit_predict(distance_matrix)

    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(entities[i])

    logging.info(f"Clustering complete. Found {len(clusters)} potential canonical groups.")
    return clusters

def generate_canonical_map(clusters: dict[int, list[str]]) -> dict[str, str]:
    """
    Genera la mappa finale scegliendo il nome più lungo (e alfabeticamente primo in caso di parità) come canonico.
    """
    canonical_map = {}
    logging.info("Generating canonical map...")
    for cluster_id, members in clusters.items():
        if len(members) > 1:
            members.sort(key=lambda x: (-len(x), x))
            canonical_name = members[0]

            logging.info(f"Cluster {cluster_id}: {members} -> Canonico: '{canonical_name}'")
            for member in members:
                canonical_map[member] = canonical_name

    logging.info(f"Canonical map generated with {len(canonical_map)} entries.")
    return canonical_map

def main():
    """Flusso principale: connetti, estrai, clusterizza, salva."""
    driver = None
    try:
        creds = load_credentials()
        uri = creds.get("neo4j_uri")
        user = creds.get("neo4j_user")
        password = creds.get("neo4j_password")

        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()

        entities = fetch_unique_entities(driver)
        if not entities:
            logging.warning("No entities found in the database. Exiting.")
            return

        clusters = build_clusters(entities, SIMILARITY_THRESHOLD)
        final_map = generate_canonical_map(clusters)

        if final_map:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(final_map, f, indent=4)
            logging.info(f"Successfully saved the canonical map to '{OUTPUT_FILE}'")
        else:
            logging.info("No significant clusters found to create a map.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if driver:
            driver.close()
            logging.info("Neo4j connection closed.")

if __name__ == "__main__":
    main()