import json
import logging
from neo4j import GraphDatabase
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_credentials

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAP_FILE = "canonical_map.json"

def get_merge_plan(map_file: str) -> list[dict]:
    """Generates the merge plan from the canonical map file."""
    try:
        with open(map_file, 'r') as f:
            full_map = json.load(f)
    except FileNotFoundError:
        logging.error(f"Map file '{map_file}' not found. Cannot proceed.")
        return []

    merge_plan = [
        {"alias": alias, "canonical": canonical}
        for alias, canonical in full_map.items()
        if alias != canonical
    ]
    logging.info(f"Generated a merge plan with {len(merge_plan)} entities to consolidate.")
    return merge_plan

def execute_merge(driver, plan: list[dict]):
    """
    Executes the merge of alias nodes into canonical nodes using a robust properties configuration.
    """
    
    # Non uso combine perché può creare liste di valori se le proprietà differiscono.
    query = """
    UNWIND $plan AS row
    MATCH (alias:PoliticalEntity {name: row.alias})
    MATCH (canonical:PoliticalEntity {name: row.canonical})
    CALL apoc.refactor.mergeNodes([canonical, alias], {
        properties: {
            name: 'first', // Prendi il nome dal primo nodo della lista (il canonico)
            type: 'first'  // Anche il tipo
        }
    }) YIELD node
    RETURN canonical.name AS merged_into, row.alias AS merged_from
    """

    logging.info("Starting merge process on the database with robust property handling...")
    with driver.session(database="neo4j") as session:
        try:
            result = session.run(query, plan=plan)
            merged_count = len(list(result))
            logging.info(f"Merge operation complete. {merged_count} alias nodes were successfully merged.")
        except Exception as e:
            logging.error(f"A critical error occurred during the merge transaction: {e}")
            raise

def main():
    """Main execution block for the merge script."""
    driver = None
    try:
        # Pulisce la mappa da eventuali regole errate prima di creare il piano
        with open(MAP_FILE, 'r+') as f:
            data = json.load(f)
            if "Article 2" in data and data["Article 2"] == "Article 1":
                del data["Article 2"]
                logging.warning("Removed incorrect mapping for 'Article 2'.")
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()

        plan = get_merge_plan(MAP_FILE)
        if not plan:
            logging.info("No entities to merge. Exiting transformation phase.")
            return

        creds = load_credentials()
        uri, user, password = creds.get("neo4j_uri"), creds.get("neo4j_user"), creds.get("neo4j_password")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        execute_merge(driver, plan)

    except Exception as e:
        logging.error(f"Failed to complete the merge process: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if driver:
            driver.close()

if __name__ == "__main__":
    main()