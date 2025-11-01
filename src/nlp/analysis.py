import logging
from typing import List, Tuple
import asyncio
from src.graph import schemas
from src.llm.core import entity_chain, stance_chain, stance_chain_contextual


class NLPProcessingError(Exception):
    """Eccezione custom per fallimenti nel processing NLP."""
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- MODELLI DI EMBEDDING ---
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

# --- CARICAMENTO MODELLI ---
def _load_embedding_model():
    """Carica il modello di embedding con gestione robusta degli errori."""
    try:
        logging.info("Loading embedding model: sentence-transformers/all-mpnet-base-v2")
        return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    except Exception as e:
        logging.error(f"FATAL: Failed to load embedding model. Error: {e}")
        raise RuntimeError("Embedding model initialization failed") from e

embedding_model = _load_embedding_model()
BATCH_SIZE = 32

# --- CONTROLLO DELLA CONCORRENZA ---
# Limita il numero di chiamate LLM concorrenti per evitare crash
LLM_CONCURRENCY_LIMIT = 5
llm_semaphore = None  # Inizializza a None

async def _get_llm_semaphore():
    """Inizializza il semaforo se non esiste."""
    global llm_semaphore
    if llm_semaphore is None:
        llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY_LIMIT)
    return llm_semaphore

# --- FUNZIONI CORE ---

async def _process_with_semaphore(chain_func, *args):
    """Esegue una chiamata alla chain LLM proteggendola con un semaforo."""
    semaphore = await _get_llm_semaphore()
    async with semaphore:
        return await chain_func(*args)

async def extract_entities_from_batch(texts: List[str]) -> List[List[schemas.PoliticalEntity]]:
    """Estrae entità iterando su ogni testo e chiamando la chain singolarmente."""
    if not texts:
        return []

    all_entities: List[List[schemas.PoliticalEntity]] = []
    try:
        # Applica il semaforo a ogni chiamata
        tasks = [_process_with_semaphore(entity_chain, text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res_dict in results:
            if isinstance(res_dict, Exception):
                all_entities.append([])
                continue

            entities_from_llm = res_dict.get("entities")
            if entities_from_llm is None or not isinstance(entities_from_llm, list):
                all_entities.append([])
                continue

            valid_entities = [
                schemas.PoliticalEntity(name=str(item).strip(), type="POLITICAL")
                for item in entities_from_llm if str(item).strip()
            ]
            all_entities.append(valid_entities)
        return all_entities

    except Exception as e:
        logging.error(f"Entity extraction process failed: {e}.")
        raise NLPProcessingError("Entity extraction failed") from e

async def detect_stance_from_batch(text_entity_pairs: List[Tuple[str, str]]) -> List[schemas.Stance]:
    """Determina la stance iterando su ogni coppia (testo, entità)."""
    if not text_entity_pairs:
        return []

    # Applica il semaforo a ogni chiamata
    tasks = [_process_with_semaphore(stance_chain, text, entity) for text, entity in text_entity_pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_stances: List[schemas.Stance] = []
    for i, res_dict in enumerate(results):
        text, entity = text_entity_pairs[i]
        if isinstance(res_dict, Exception) or "stance" not in res_dict:
            continue # Salta il fallimento
        
        stance_label = str(res_dict.get("stance", "NEUTRAL")).upper().strip()
        confidence = float(res_dict.get("confidence", 0.0))

        all_stances.append(
            schemas.Stance(
                target_entity_name=entity,
                stance=stance_label,
                confidence=confidence,
                sentence=text,
            )
        )
    return all_stances

async def detect_stance_from_batch_contextual(contextual_pairs: List[Tuple[str, str, str]]) -> List[schemas.Stance]:
    """Determina la stance per i commenti usando il contesto del post."""
    if not contextual_pairs:
        return []

    # Applica il semaforo a ogni chiamata
    tasks = [_process_with_semaphore(stance_chain_contextual, post_content, comment_content, entity) for post_content, comment_content, entity in contextual_pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_stances: List[schemas.Stance] = []
    for i, res_dict in enumerate(results):
        _, comment_content, entity = contextual_pairs[i]
        if isinstance(res_dict, Exception) or "stance" not in res_dict:
            continue
        
        stance_label = str(res_dict.get("stance", "NEUTRAL")).upper().strip()
        confidence = float(res_dict.get("confidence", 0.0))

        all_stances.append(
            schemas.Stance(
                target_entity_name=entity,
                stance=stance_label,
                confidence=confidence,
                sentence=comment_content,
            )
        )
    return all_stances


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def generate_embedding_batch(texts: List[str]) -> List[List[float]]:
    if not texts: return []
    try:
        embeddings = embedding_model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_tensor=False)
        return embeddings.tolist()
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        raise RuntimeError("Embedding generation failed") from e