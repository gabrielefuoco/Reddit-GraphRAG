import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
import asyncio
from neo4j import Driver, GraphDatabase
import spacy
from spacy.language import Language
import ollama
import httpx
from src.data_processing.cleaner import clean_text
from src.nlp.analysis import (
    generate_embedding_batch,
    extract_entities_from_batch,
    detect_stance_from_batch,
    detect_stance_from_batch_contextual,
    NLPProcessingError,
)
from src.graph.schemas import Post, Comment
from src.data_processing.ingestion import fetch_reddit_data
from src.utils.config import load_credentials
from pydantic import ValidationError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class GraphBuilder:
    """Orchestrates the ETL pipeline to process raw data and load it into Neo4j."""

    def __init__(self, driver: Driver, nlp: Language):
        self.driver = driver
        self.nlp = nlp
        logging.info("GraphBuilder initialized.")

    async def _check_ollama_health(self):
        logging.info("Performing health check on Ollama service...")
        try:
            client = ollama.AsyncClient(timeout=10)
            await client.list()
            logging.info("Ollama service is responsive.")
        except (ollama.ResponseError, httpx.ConnectError, httpx.ReadTimeout) as e:
            logging.error(f"FATAL: Ollama service is unavailable. Halting pipeline. Error: {e}")
            raise ConnectionError("Ollama service is unavailable.") from e

    def _save_failed_item_to_dlq(self, failed_item: Dict[str, Any], reason: str, item_type: str):
        dlq_dir = f"data/failed_{item_type}s"
        os.makedirs(dlq_dir, exist_ok=True)
        item_id = failed_item.get("id", "unknown_id")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = os.path.join(dlq_dir, f"failed_{item_type}_{item_id}_{timestamp}.json")

        with open(file_path, "w") as f:
            json.dump({"reason": reason, "item_data": failed_item}, f, indent=4)
        logging.warning(f"Saved failed {item_type} {item_id} to DLQ: {file_path}")

    async def run_etl_pipeline(
        self, raw_data: List[Tuple[Dict, List[Dict]]], mini_batch_size: int = 10
    ):
        """Runs the full ETL pipeline on batches of (post, comments_list)."""
        await self._check_ollama_health()
        if not raw_data:
            logging.warning("Received an empty list of data. Nothing to process.")
            return

        total_items = len(raw_data)
        logging.info(
            f"Starting ETL pipeline for {total_items} posts (and their comments) with mini-batch size of {mini_batch_size}."
        )

        for i in range(0, total_items, mini_batch_size):
            mini_batch = raw_data[i : i + mini_batch_size]
            logging.info(f"Processing mini-batch {i//mini_batch_size + 1}/{(total_items + mini_batch_size - 1)//mini_batch_size}...")
            
            enriched_posts, enriched_comments = await self._process_mini_batch_parallel(mini_batch)
            
            if enriched_posts or enriched_comments:
                self._load_data(enriched_posts, enriched_comments)

        logging.info("ETL pipeline run completed.")

    async def _process_mini_batch_parallel(
        self, mini_batch: List[Tuple[Dict, List[Dict]]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Processes posts and their comments in parallel with robust error handling."""
        enriched_posts, enriched_comments = [], []
        
        try:
            posts_raw = [item[0] for item in mini_batch]
            comments_raw = [comment for item in mini_batch for comment in item[1]]

            all_raw_contents = [p.get('content', '') for p in posts_raw] + [c.get('content', '') for c in comments_raw]
            all_cleaned_texts = [clean_text(text, self.nlp) for text in all_raw_contents]
            
            all_embeddings = generate_embedding_batch(all_cleaned_texts)
            
            all_entities_results = await extract_entities_from_batch(all_cleaned_texts)

            # Splitta risultati per post e commenti
            post_cleaned_texts = all_cleaned_texts[:len(posts_raw)]
            comment_cleaned_texts = all_cleaned_texts[len(posts_raw):]
            post_embeddings = all_embeddings[:len(posts_raw)]
            comment_embeddings = all_embeddings[len(posts_raw):]
            post_entities_lists = all_entities_results[:len(posts_raw)]
            comment_entities_lists = all_entities_results[len(posts_raw):]

            # --- Stance Detection ---
            post_stance_pairs = [(posts_raw[i]['content'], e.name) for i, entities in enumerate(post_entities_lists) for e in entities if posts_raw[i].get('content')]
            post_stances_flat = await detect_stance_from_batch(post_stance_pairs)
            
            post_content_map = {p['id']: p.get('content', '') for p in posts_raw}
            comment_stance_pairs = []
            for i, entities in enumerate(comment_entities_lists):
                comment = comments_raw[i]
                post_content = post_content_map.get(comment['post_id'], "")
                if not comment.get('content'): continue
                for entity in entities:
                    comment_stance_pairs.append((post_content, comment['content'], entity.name))
            comment_stances_flat = await detect_stance_from_batch_contextual(comment_stance_pairs)

            # --- Assemblaggio dei dati arricchiti ---
            for i, post_data in enumerate(posts_raw):
                try:
                    post_obj = Post(
                        id=post_data["id"],
                        author=post_data.get("author", "deleted"),
                        content=post_data.get('content', ''),
                        cleaned_content=post_cleaned_texts[i],
                        timestamp=post_data["timestamp"],
                        score=post_data["score"],
                        subreddit=post_data["subreddit"],
                        entities=post_entities_lists[i],
                        stances=[s for s in post_stances_flat if s.sentence == post_data.get('content', '')],
                        embedding=post_embeddings[i],
                    )
                    enriched_posts.append(post_obj.model_dump())
                except (ValidationError, Exception) as e:
                    self._save_failed_item_to_dlq(post_data, f"Post assembly failed: {e}", "post")

            for i, comment_data in enumerate(comments_raw):
                try:
                    comment_obj = Comment(
                        id=comment_data["id"],
                        post_id=comment_data["post_id"],
                        author=comment_data.get("author", "deleted"),
                        content=comment_data.get('content', ''),
                        cleaned_content=comment_cleaned_texts[i],
                        timestamp=comment_data["timestamp"],
                        score=comment_data["score"],
                        entities=comment_entities_lists[i],
                        stances=[s for s in comment_stances_flat if s.sentence == comment_data.get('content', '')],
                        embedding=comment_embeddings[i],
                    )
                    enriched_comments.append(comment_obj.model_dump())
                except (ValidationError, Exception) as e:
                    self._save_failed_item_to_dlq(comment_data, f"Comment assembly failed: {e}", "comment")

        except NLPProcessingError as e:
            logging.error(f"A critical NLP error occurred during batch processing: {e}. Skipping batch.")
        except Exception as e:
            logging.error(f"An unexpected error occurred during batch processing: {e}. Skipping batch.")

        return enriched_posts, enriched_comments

    def _load_data(self, posts_batch: List[Dict], comments_batch: List[Dict]):
        """Loads posts and comments into Neo4j using a single, robust transaction."""
        
        cypher_query = """
        // Phase 1: Ingest Posts
        UNWIND $posts AS post_data
        MERGE (p:Post {id: post_data.id})
        SET p.content = post_data.content,
            p.cleaned_content = post_data.cleaned_content,
            p.timestamp = post_data.timestamp,
            p.score = post_data.score,
            p.subreddit = post_data.subreddit,
            p.embedding = post_data.embedding
        
        FOREACH (_ IN CASE WHEN post_data.author <> "deleted" THEN [1] ELSE [] END |
            MERGE (u:User {name: post_data.author})
            MERGE (u)-[:POSTED]->(p)
        )
        FOREACH (entity_data IN post_data.entities |
            MERGE (e:PoliticalEntity {name: entity_data.name})
            ON CREATE SET e.type = entity_data.type
            MERGE (p)-[:MENTIONS]->(e)
        )
        FOREACH (stance_data IN post_data.stances |
            MERGE (e_stance:PoliticalEntity {name: stance_data.target_entity_name})
            MERGE (p)-[r:HAS_STANCE]->(e_stance)
            SET r.stance = stance_data.stance, r.confidence = stance_data.confidence
        )
        
        WITH 1 as placeholder

        // Phase 2: Ingest Comments
        UNWIND $comments AS comment_data
        MATCH (parent_post:Post {id: comment_data.post_id})
        MERGE (c:Comment {id: comment_data.id})
        SET c.content = comment_data.content,
            c.cleaned_content = comment_data.cleaned_content,
            c.timestamp = comment_data.timestamp,
            c.score = comment_data.score,
            c.embedding = comment_data.embedding
        
        MERGE (c)-[:REPLY_TO]->(parent_post)

        FOREACH (_ IN CASE WHEN comment_data.author <> "deleted" THEN [1] ELSE [] END |
            MERGE (u:User {name: comment_data.author})
            MERGE (u)-[:POSTED]->(c)
        )
        FOREACH (entity_data IN comment_data.entities |
            MERGE (e:PoliticalEntity {name: entity_data.name})
            ON CREATE SET e.type = entity_data.type
            MERGE (c)-[:MENTIONS]->(e)
        )
        FOREACH (stance_data IN comment_data.stances |
            MERGE (e_stance:PoliticalEntity {name: stance_data.target_entity_name})
            MERGE (c)-[r:HAS_STANCE]->(e_stance)
            SET r.stance = stance_data.stance, r.confidence = stance_data.confidence
        )
        """
        try:
            with self.driver.session(database="neo4j") as session:
                session.run(cypher_query, posts=posts_batch, comments=comments_batch)
            logging.info(
                f"Successfully loaded batch of {len(posts_batch)} posts and {len(comments_batch)} comments into Neo4j."
            )
        except Exception as e:
            logging.error(f"Failed to load data into Neo4j: {e}")
            for post in posts_batch:
                self._save_failed_item_to_dlq(post, f"Neo4j transaction failed: {e}", "post")
            for comment in comments_batch:
                self._save_failed_item_to_dlq(comment, f"Neo4j transaction failed: {e}", "comment")

async def main():
    """Main execution function to run the ETL pipeline."""
    SUBREDDITS = [
    "politics",
    "PoliticalDiscussion",
    "Conservative",
    "Liberal",
    "antiwork",
    "changemyview",
]

    POST_LIMIT = 250

    driver = None
    try:
        logging.info("--- STARTING ETL PROCESS ---")
        creds = load_credentials()
        uri, user, password = creds.get("neo4j_uri"), creds.get("neo4j_user"), creds.get("neo4j_password")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        nlp = spacy.load("en_core_web_sm")
        graph_builder = GraphBuilder(driver=driver, nlp=nlp)

        logging.info(f"Fetching {POST_LIMIT} posts and their comments from subreddits: {', '.join(SUBREDDITS)}...")
        
        raw_data = [item async for item in fetch_reddit_data(SUBREDDITS, limit=POST_LIMIT)]
            
        logging.info(f"Successfully fetched data for {len(raw_data)} posts.")

        if raw_data:
            await graph_builder.run_etl_pipeline(raw_data)
        else:
            logging.warning("No data was fetched. ETL pipeline will not run.")

        logging.info("--- ETL PROCESS FINISHED ---")

    except Exception as e:
        logging.error(f"An error occurred during the main execution: {e}", exc_info=True)
    finally:
        if driver:
            driver.close()

if __name__ == "__main__":
    asyncio.run(main())