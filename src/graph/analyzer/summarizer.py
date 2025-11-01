import logging
import asyncio
from typing import List, Dict, Any
from neo4j import Driver
from src.nlp.analysis import generate_embedding_batch
from src.llm.core import summary_chain

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class IdeologySummarizer:
    """
    Summarizes ideologies based on (PoliticalEntity, Stance) pairs,
    using both posts and their aligned comments as context.
    """

    def __init__(self, driver: Driver):
        self.driver = driver
        # PARAMETRI DI CONTROLLO
        self.MIN_POSTS_FOR_SUMMARY = 3
        self.TOP_K_POSTS = 10
        self.TOP_N_COMMENTS = 5
        self.CONFIDENCE_THRESHOLD = 0.85

    async def _get_target_ideologies(self) -> List[Dict[str, str]]:
        """
        Identifies all unique (PoliticalEntity, Stance) pairs that have enough high-confidence
        posts to be worth summarizing.
        """
        logging.info("Identificazione delle ideologie target da riassumere...")
        query = """
        MATCH (p:Post)-[r:HAS_STANCE]->(e:PoliticalEntity)
        WHERE r.confidence >= $threshold
        WITH e.name AS entityName, r.stance AS stance, count(p) AS postCount
        WHERE postCount >= $min_posts
        RETURN entityName, stance
        ORDER BY entityName, stance
        """
        with self.driver.session(database="neo4j") as session:
            results = session.run(
                query,
                threshold=self.CONFIDENCE_THRESHOLD,
                min_posts=self.MIN_POSTS_FOR_SUMMARY
            ).data()
        logging.info(f"Trovate {len(results)} ideologie target da analizzare.")
        return results

    async def _build_context_dossier(self, entity_name: str, stance: str) -> str:
        """
        Constructs a rich text dossier for a given ideology, combining top posts
        with their most relevant, stance-aligned comments.
        """
        logging.info(f"Costruzione dossier per l'ideologia: {entity_name} ({stance})")
        # --- INIZIO QUERY ---
        query = """
        // Fase 1: Trova i post "seme" più rilevanti per l'ideologia
        MATCH (p:Post)-[r:HAS_STANCE]->(e:PoliticalEntity {name: $entity_name})
        WHERE r.stance = $stance AND r.confidence >= $threshold
        WITH p, e ORDER BY p.score DESC LIMIT $top_k_posts

        // Fase 2: Per ogni post seme, esegui una subquery con la sintassi moderna
        CALL {
            WITH p, e
            OPTIONAL MATCH (c:Comment)-[:REPLY_TO]->(p)
            MATCH (c)-[rc:HAS_STANCE]->(e)
            WHERE rc.stance = $stance AND rc.confidence >= $threshold
            WITH c ORDER BY c.score DESC
            RETURN COLLECT(c.content)[..$top_n_comments] AS comments
        }
        RETURN p.content AS post_content, comments AS comment_contents
        """
        # --- FINE QUERY ---

        with self.driver.session(database="neo4j") as session:
            records = session.run(
                query,
                entity_name=entity_name,
                stance=stance,
                threshold=self.CONFIDENCE_THRESHOLD,
                top_k_posts=self.TOP_K_POSTS,
                top_n_comments=self.TOP_N_COMMENTS
            ).data()

        if not records:
            return ""

        dossier_parts = []
        for record in records:
            post_text = record['post_content']
            comments = record['comment_contents']
            
            dossier_parts.append(f"POST: {post_text}")
            if comments:
                dossier_parts.append("REAZIONI DI SUPPORTO:")
                for comment in comments:
                    dossier_parts.append(f"- {comment}")
            dossier_parts.append("---")
        
        return "\n".join(dossier_parts)

    async def _summarize_and_persist(self, entity_name: str, stance: str):
        """Orchestrates the summarization for a single ideology and saves it."""
        dossier = await self._build_context_dossier(entity_name, stance)
        
        if not dossier:
            logging.warning(f"Dossier vuoto per {entity_name} ({stance}). Salto.")
            return

        summary_text = await summary_chain(posts=dossier)
        if "Unable to generate summary" in summary_text:
            logging.error(f"Fallimento generazione riassunto per {entity_name} ({stance}).")
            return

        summary_embedding_list = generate_embedding_batch([summary_text])
        if not summary_embedding_list:
             logging.error(f"Fallimento embedding per riassunto di {entity_name} ({stance}).")
             return
        summary_embedding = summary_embedding_list[0]

        summary_id = f"{entity_name}:{stance}"
        merge_query = """
        MERGE (s:IdeologicalSummary {id: $id})
        SET s.summary = $summary, 
            s.embedding = $embedding,
            s.stance = $stance
        WITH s
        MATCH (e:PoliticalEntity {name: $entity_name})
        MERGE (s)-[:SUMMARIZES_STANCE_ON]->(e)
        """
        with self.driver.session(database="neo4j") as session:
            session.run(
                merge_query,
                id=summary_id,
                summary=summary_text,
                embedding=summary_embedding,
                stance=stance,
                entity_name=entity_name
            )
        logging.info(f"Riassunto per {entity_name} ({stance}) salvato con successo.")

    async def summarize_ideologies(self):
        """Main entry point to summarize all discoverable ideologies."""
        target_ideologies = await self._get_target_ideologies()

        if not target_ideologies:
            logging.info("Nessuna ideologia sufficientemente supportata trovata per la riassunzione.")
            return

        # --- LOGICA DI BATCHING PER EVITARE SOVRACCARICO ---
        BATCH_SIZE = 5  
        logging.info(f"Processing {len(target_ideologies)} ideologies in batches of {BATCH_SIZE}...")

        for i in range(0, len(target_ideologies), BATCH_SIZE):
            batch = target_ideologies[i:i + BATCH_SIZE]
            logging.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(target_ideologies) + BATCH_SIZE - 1)//BATCH_SIZE}...")
            
            tasks = [
                self._summarize_and_persist(ideo['entityName'], ideo['stance'])
                for ideo in batch
            ]
            await asyncio.gather(*tasks)
            
            logging.info(f"Batch {i//BATCH_SIZE + 1} completed. Pausing briefly...")
            await asyncio.sleep(2)  
        # --- FINE LOGICA DI BATCHING ---

        logging.info("Processo di riassunto delle ideologie completato.")