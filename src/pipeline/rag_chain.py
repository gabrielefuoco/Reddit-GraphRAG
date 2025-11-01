
import logging
from typing import Dict, Any, List, Optional
import neo4j
import asyncio
from src.nlp.analysis import generate_embedding_batch
from src.llm.core import rag_chain, entity_chain, stance_chain
from src.graph.schemas import PoliticalEntity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GraphRAGPipeline:
    def __init__(self, driver: neo4j.Driver):
        if not driver or not isinstance(driver, neo4j.Driver):
            raise ValueError("A valid neo4j.Driver instance is required.")
        
        self.driver = driver
        logging.info("GraphRAGPipeline initialized successfully.")

    async def _detect_stance_intent_async(self, user_query: str) -> Optional[str]:
        """
        Determina l'intento di stance della query riutilizzando le chain esistenti.
        Prima estrae un'entità, poi classifica la stance della query verso quella entità.
        """
        logging.info("Rilevamento intento tramite riciclo di chain esistenti...")
        
        # 1. Ricicla entity_chain per trovare un target
        entities_result = await entity_chain(user_query)
        entity_names = entities_result.get("entities", [])
        
        if not entity_names:
            logging.info("Nessuna entità trovata nella query. Intento considerato NEUTRAL.")
            return None
            
        # Usa solo la prima entità trovata 
        target_entity = entity_names[0]
        logging.info(f"Entità pivot trovata: '{target_entity}'. Classifico la stance della query...")

        # 2. Ricicla stance_chain per classificare la query
        stance_result = await stance_chain(text=user_query, entity=target_entity)
        stance = stance_result.get("stance", "NEUTRAL").upper()

        if stance in ["FAVORABLE", "AGAINST"]:
            logging.info(f"Rilevato intento di stance: {stance}")
            return stance
        
        logging.info("La stance della query è NEUTRAL. Nessun intento specifico rilevato.")
        return None

    def _hierarchical_retrieval(self, entity_names: list[str], stance_intent: Optional[str], top_k: int = 10) -> dict:
        logging.info(f"Avvio recupero gerarchico per entità: {entity_names}, con intento stance: {stance_intent}")
        
        summary_query = """
            UNWIND $entity_names AS entity_name
            MATCH (e:PoliticalEntity {name: entity_name})
            MATCH (s:IdeologicalSummary)-[:SUMMARIZES_STANCE_ON]->(e)
            WHERE $stance_intent IS NULL OR s.stance = $stance_intent
            RETURN s.summary AS summary, s.id AS id, s.stance as stance
            LIMIT $top_k
        """

        post_query = """
            UNWIND $entity_names AS entity_name
            CALL db.index.fulltext.queryNodes("entity_names_ft", entity_name + "~") YIELD node AS e
            MATCH (p:Post)-[r:HAS_STANCE]->(e)
            WHERE $stance_intent IS NULL OR r.stance = $stance_intent
            RETURN p.content AS text, p.id AS id, r.stance AS stance
            ORDER BY p.score DESC
            LIMIT $top_k
        """
                
        context = {"summaries": [], "posts": []}
        try:
            with self.driver.session(database="neo4j") as session:
                summary_results = session.run(summary_query, entity_names=entity_names, stance_intent=stance_intent, top_k=top_k)
                context["summaries"] = [record.data() for record in summary_results]
                logging.info(f"Recuperati {len(context['summaries'])} riassunti ideologici.")
                post_results = session.run(post_query, entity_names=entity_names, stance_intent=stance_intent, top_k=top_k)                
                context["posts"] = [record.data() for record in post_results]
                logging.info(f"Recuperati {len(context['posts'])} post.")
        except Exception as e:
            logging.error(f"Errore durante il recupero gerarchico: {e}")

        return context

    def _semantic_fallback_retrieval(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        cypher_query = """
            CALL db.index.vector.queryNodes('post_embedding', $top_k, $embedding)
            YIELD node, score
            RETURN node.content AS text, node.id AS id, score
        """
        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(cypher_query, top_k=top_k, embedding=query_embedding)
                return result.data()
        except Exception as e:
            logging.error(f"Errore durante il recupero semantico di fallback: {e}")
            return []

    async def _generate_response_async(self, context_data: dict, user_query: str) -> str:
        formatted_context = "### Riassunti delle Prospettive Ideologiche Rilevanti\n"
        if context_data.get("summaries"):
            for s in context_data["summaries"]:
                formatted_context += f"- (Prospettiva {s['stance']} su {s['id'].split(':')[0]}): {s['summary']}\n"
        else:
            formatted_context += "Nessun riassunto rilevante trovato.\n"

        formatted_context += "\n### Esempi Specifici da Post Individuali\n"
        if context_data.get("posts"):
            for p in context_data["posts"]:
                text_content = p.get('text', '').strip()
                if text_content:
                    formatted_context += f"- (Post {p.get('id', 'N/A')}, Stance: {p.get('stance', 'N/A')}): {text_content}\n"
        else:
            formatted_context += "Nessun post individuale rilevante trovato.\n"
        
        return await rag_chain(context=formatted_context, user_query=user_query)

    # --- LOGICA PRINCIPALE ASINCRONA ---
    async def query_async(self, user_query: str, top_k: int = 30) -> Dict[str, Any]:
        logging.info(f"Ricevuta query: '{user_query}'")
        
        entities_result = await entity_chain(user_query)
        entity_names = entities_result.get("entities", [])
        
        stance_intent = await self._detect_stance_intent_async(user_query)
        
        embedding_list = generate_embedding_batch([user_query])
        if not embedding_list:
             return {"answer": "Impossibile generare un embedding per la query.", "context": {}, "match_type": "error"}
        query_embedding = embedding_list[0]
        
        context_data = {}
        match_type = 'none'

        if entity_names:
            context_data = self._hierarchical_retrieval(entity_names, stance_intent, top_k)
            if context_data.get("summaries") or context_data.get("posts"):
                match_type = 'hierarchical_stance_aware' if stance_intent else 'hierarchical'
        
        if not (context_data.get("summaries") or context_data.get("posts")):
            logging.info("Fallback semantico.")
            fallback_posts = self._semantic_fallback_retrieval(query_embedding, top_k)
            if fallback_posts:
                context_data = {"summaries": [], "posts": fallback_posts}
                match_type = 'semantic_fallback'

        if not context_data:
            return {"answer": "Non ho trovato informazioni.", "context": {}, "match_type": "none"}
            
        answer = await self._generate_response_async(context_data, user_query)
        return {"answer": answer, "context": context_data, "match_type": match_type}

    # Wrapper sincrono per l'agente ReAct
    def query(self, user_query: str, top_k: int = 30) -> Dict[str, Any]:
        return asyncio.run(self.query_async(user_query, top_k))