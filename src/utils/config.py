from dotenv import load_dotenv
import os

def load_credentials():
    """Loads credentials from .env file."""
    load_dotenv()
    return {
        "reddit_client_id": os.getenv("REDDIT_CLIENT_ID"),
        "reddit_client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
        "reddit_user_agent": os.getenv("REDDIT_USER_AGENT"),
        "neo4j_uri": os.getenv("NEO4J_URI"),
        "neo4j_user": os.getenv("NEO4J_USER"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD"),
    }