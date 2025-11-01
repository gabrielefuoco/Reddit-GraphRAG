import logging
from neo4j import GraphDatabase, exceptions
from src.utils.config import load_credentials

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_schema(driver):
    """
    Applies all schema constraints and indices to the Neo4j database.
    This includes Posts, Users, Entities, and the derived IdeologicalSummary nodes.
    Operations are idempotent due to 'IF NOT EXISTS'.
    """
    logging.info("Starting comprehensive schema setup...")

    schema_queries = [
        # --- Vincoli di base ---
        "CREATE CONSTRAINT post_id_unique IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT comment_id_unique IF NOT EXISTS FOR (c:Comment) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT user_name_unique IF NOT EXISTS FOR (u:User) REQUIRE u.name IS UNIQUE",
        "CREATE CONSTRAINT political_entity_name_unique IF NOT EXISTS FOR (e:PoliticalEntity) REQUIRE e.name IS UNIQUE",
        
        # --- Vincolo per i riassunti ideologici ---
        "CREATE CONSTRAINT ideological_summary_id_unique IF NOT EXISTS FOR (s:IdeologicalSummary) REQUIRE s.id IS UNIQUE",

        # --- Indice Full-Text per ricerca ibrida ---
        "CREATE FULLTEXT INDEX entity_names_ft IF NOT EXISTS FOR (n:PoliticalEntity) ON EACH [n.name]",
        
        # --- Indici Vettoriali per ricerca semantica ---
        """
        CREATE VECTOR INDEX post_embedding IF NOT EXISTS FOR (p:Post) ON (p.embedding)
        OPTIONS { indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' } }
        """,
        """
        CREATE VECTOR INDEX comment_embedding IF NOT EXISTS FOR (c:Comment) ON (c.embedding)
        OPTIONS { indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' } }
        """,
        # --- Indice vettoriale per i riassunti ---
        """
        CREATE VECTOR INDEX ideological_summary_embedding IF NOT EXISTS FOR (s:IdeologicalSummary) ON (s.embedding)
        OPTIONS { indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' } }
        """
    ]

    try:
        with driver.session() as session:
            for query in schema_queries:
                logging.info(f"Applying schema rule: {query.strip().splitlines()[0]}...")
                session.run(query)
                logging.info("Rule applied successfully.")
        logging.info("Schema setup completed successfully.")
    except exceptions.Neo4jError as e:
        logging.error(f"An error occurred during schema setup: {e}")
        raise

def main():
    """
    Main execution block to set up the database schema.
    """
    credentials = None
    driver = None
    try:
        logging.info("Loading database credentials...")
        credentials = load_credentials()
        uri = credentials.get("neo4j_uri")
        user = credentials.get("neo4j_user")
        password = credentials.get("neo4j_password")

        if not all([uri, user, password]):
            logging.error("Missing Neo4j credentials in the environment file.")
            return

        logging.info("Connecting to Neo4j database...")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logging.info("Database connection successful.")

        setup_schema(driver)

    except exceptions.ServiceUnavailable as e:
        logging.error(f"Could not connect to Neo4j at {credentials.get('neo4j_uri', 'N/A')}. "
                      f"Please ensure the database is running and credentials are correct. Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if driver:
            driver.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()