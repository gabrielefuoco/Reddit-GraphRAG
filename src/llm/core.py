import logging
from enum import Enum
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, validator
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


llm_json = ChatOllama(
    model="qwen3:4b-instruct-2507-q4_K_M",
    format="json",
    temperature=0.0,  # Deterministico per extraction/classification
    timeout=30  
)

llm_text = ChatOllama(
    #model="qwen3:4b-instruct-2507-q4_K_M",
    model="qwen3:4b-instruct-2507-q8_0",
    temperature=0.0,  # Deterministico altrimenti l'agente non segue bene le istruzioni
    timeout=180
)

# Parser per risposte testuali semplici
string_parser = StrOutputParser()

class EntitiesOutput(BaseModel):
    entities: List[str] = Field(description="List of extracted political entities")
    @validator('entities')
    def normalize_entities(cls, v):
        if not v: return []
        seen = set()
        normalized = []
        for entity in v:
            entity_clean = entity.strip()
            entity_lower = entity_clean.lower()
            if entity_lower and entity_lower not in seen:
                seen.add(entity_lower)
                normalized.append(entity_clean)
        return normalized

class StanceEnum(str, Enum):
    FAVORABLE = "FAVORABLE"
    AGAINST = "AGAINST"
    NEUTRAL = "NEUTRAL"

class StanceOutput(BaseModel):
    stance: StanceEnum = Field(description="The detected stance")
    confidence: float = Field(description="Confidence score", ge=0.0, le=1.0)

entity_parser = JsonOutputParser(pydantic_object=EntitiesOutput)
stance_parser = JsonOutputParser(pydantic_object=StanceOutput)


# --- Definizione dei PROMPT ---


entity_prompt_template = """You are a high-precision Named Entity Recognition model. Your ONLY task is to identify key political figures, organizations, or specific political concepts from the user's text.

Follow these rules STRICTLY:
1. Respond ONLY with a valid JSON object. Do not add any text, markdown, or comments before or after the JSON.
2. The JSON object must have a single key: "entities".
3. The value of "entities" must be a list of strings.
4. If no relevant entities are found, the list must be empty.
5. Extract entities in their most common form (e.g., "Biden" for "Joe Biden", "President Biden").
6. Avoid extracting generic terms like "politics", "economy", "government" unless they refer to specific entities.

---
EXAMPLES:

Text: "Qual è l'opinione generale su Joe Biden?"
JSON Output:
{{
  "entities": ["Joe Biden"]
}}

Text: "Cosa pensano del Partito Democratico e delle politiche di Trump?"
JSON Output:
{{
  "entities": ["Partito Democratico", "Trump"]
}}

Text: "parlami di economia"
JSON Output:
{{
  "entities": []
}}
---

Text: "{text}"
JSON Output:
"""
entity_prompt = ChatPromptTemplate.from_template(entity_prompt_template)


stance_prompt_template = """You are a Stance Classifier. Your task is to determine the stance of a text towards "{entity}" as FAVORABLE, AGAINST, or NEUTRAL.

Classification Rules:
- FAVORABLE: The text expresses positive sentiment, support, or approval towards the entity
- AGAINST: The text expresses negative sentiment, criticism, or disapproval towards the entity
- NEUTRAL: Use ONLY for purely factual statements without opinion indicators.

If the text contains any opinionated language, you must classify it as FAVORABLE or AGAINST.

---
Text: "{text}"

Output a JSON object with "stance" and "confidence" keys:
"""
stance_prompt = ChatPromptTemplate.from_template(stance_prompt_template)

# --- PROMPT CONTESTUALE PER COMMENTI ---
stance_contextual_prompt_template = """You are a high-precision contextual stance detection analyst. Your task is to determine the stance of a COMMENT in relation to the ORIGINAL POST it is replying to.

### CONTEXT ###
ORIGINAL POST (for context only):
"{post_content}"

### TARGET FOR ANALYSIS ###
COMMENT:
"{comment_content}"

### ENTITY TO ANALYZE ###
"{entity}"

### INSTRUCTIONS ###
1.  Read the ORIGINAL POST to understand the topic of discussion.
2.  Analyze the COMMENT to determine its stance (FAVORABLE, AGAINST, or NEUTRAL) strictly towards the specified ENTITY.
3.  Use the post's context to interpret sarcasm, irony, or implicit references within the comment. The stance of the post itself is irrelevant.
4.  Your output MUST be a single, valid JSON object with two keys: "stance" (one of "FAVORABLE", "AGAINST", "NEUTRAL") and "confidence" (a float between 0.0 and 1.0).

JSON OUTPUT:
"""
stance_contextual_prompt = ChatPromptTemplate.from_template(stance_contextual_prompt_template)

summary_prompt_template = """You are an expert political analyst and narrative summarizer. Your task is to produce a **comprehensive, well-structured summary** that captures the full meaning, tone, and nuances of a collection of political posts.

### OBJECTIVE ###
Write a detailed, insightful synthesis that reflects:
- The **main political topics** and subthemes discussed.
- The **prevailing opinions, arguments, and sentiments** (positive, negative, polarized, etc.).
- Any **recurring expressions or notable quotes** that exemplify the overall discourse.
- The **tone and rhetorical style** (e.g., emotional, sarcastic, factual, ideological).
- When relevant, note **contrasting viewpoints** or **areas of consensus/conflict**.

### WRITING STYLE ###
- Write in **fluent, natural Italian** as a professional political journalist or analyst.
- The summary should be **rich and articulated** (typically 3–6 well-developed paragraphs).
- Maintain a **neutral and analytical tone** — describe opinions without endorsing them.
- You **may integrate short quotes or key expressions** from the posts *if they add insight*.
- Avoid bullet points, introductions like "In summary," or any metatextual remarks.
- Do **not** refer to “posts” or “users” explicitly — treat the material as a single discourse body.

### STRUCTURE (recommended but flexible) ###
1. **Overview:** Introduce the general context or theme emerging from the posts.  
2. **Key Themes:** Describe the main political or ideological issues being debated.  
3. **Sentiment & Tone:** Analyze the emotional charge, rhetoric, and stance distribution.  
4. **Notable Perspectives or Quotes:** Highlight emblematic phrases or contrasting arguments.  
5. **Concluding Insight:** End with a balanced synthesis of the collective attitude or outlook.

---
POSTS:
{posts}
---

DETAILED SUMMARY (in Italian):
"""


summary_prompt = ChatPromptTemplate.from_template(summary_prompt_template)

rag_prompt_template = """You are a political analyst AI. Your sole function is to answer the user's question using ONLY the provided context.

**Internal Thought Process (DO NOT display this in your output):**
1. Carefully review all provided documents and identify the most relevant, information-rich portions ('signal') that directly relate to the user's question.
2. Extract and synthesize key facts, perspectives, and quotations from those portions to support a comprehensive, nuanced answer.
3. If multiple viewpoints or interpretations appear, summarize them clearly and fairly.
4. If no relevant information is found or the evidence is insufficient, output the mandatory failure message and nothing else.

**User-Facing Output Rules:**
- Your response must be written in **natural, fluent language**, using user's language, in the style of a **professional political analyst or journalist**.
- Provide a **complete and detailed explanation**, not a summary — expand on causes, implications, and contextual elements when possible.
- When relevant, **include short, verbatim quotes** or key expressions from the context to strengthen the answer (integrated smoothly into the text, not as bullet points or citations).
- Avoid mentioning the context, the documents, or your reasoning process.
- Do NOT invent or assume facts not present in the provided material.
- If you cannot answer, your entire response must be EXACTLY: "Non ho abbastanza informazioni nel contesto fornito per rispondere a questa domanda."

---
**CONTESTO:**
{context}

**DOMANDA:**
{user_query}

---
**FINAL ANSWER (in user language):**
"""

rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

# ... Costruzione delle chain  ...

def create_async_chain_wrapper(base_chain, default_response):
    """Factory to create robust async wrapper functions for different chains."""
    async def safe_ainvoke(input_data: dict, max_retries: int = 2) -> any:
        for attempt in range(max_retries):
            try:
                result = await base_chain.ainvoke(input_data, config=RunnableConfig(max_concurrency=1))
                return result
            except Exception as e:
                logger.error(f"Chain invocation failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.warning(f"All retry attempts exhausted, returning default response: {default_response}")
                    return default_response
        return default_response
    return safe_ainvoke

# Create wrappers
entity_chain_async_wrapper = create_async_chain_wrapper(entity_prompt | llm_json | entity_parser, {"entities": []})
stance_chain_async_wrapper = create_async_chain_wrapper(stance_prompt | llm_json | stance_parser, {"stance": "NEUTRAL", "confidence": 0.0})
stance_contextual_chain_async_wrapper = create_async_chain_wrapper(stance_contextual_prompt | llm_json | stance_parser, {"stance": "NEUTRAL", "confidence": 0.0})
summary_chain_async_wrapper = create_async_chain_wrapper(summary_prompt | llm_text | string_parser, "Unable to generate summary.")
rag_chain_async_wrapper = create_async_chain_wrapper(rag_prompt | llm_text | string_parser, "I could not find an answer.")

# Esporta le funzioni async per l'uso esterno
async def entity_chain(text: str):
    return await entity_chain_async_wrapper({"text": text})

async def stance_chain(text: str, entity: str):
    return await stance_chain_async_wrapper({"text": text, "entity": entity})

async def stance_chain_contextual(post_content: str, comment_content: str, entity: str):
    # Tronca il contesto del post per sicurezza ####################
    # ################################# RICORDARSI DI TESTARE CON VALORI PIÙ GRANDI
    MAX_CONTEXT_CHARS = 1000
    if len(post_content) > MAX_CONTEXT_CHARS:
        post_content = post_content[:MAX_CONTEXT_CHARS] + "..."
    return await stance_contextual_chain_async_wrapper({
        "post_content": post_content,
        "comment_content": comment_content,
        "entity": entity
    })

async def summary_chain(posts: str):
    MAX_CHARS = 24000
    if len(posts) > MAX_CHARS:
        logger.warning(f"Posts too long ({len(posts)} chars), truncating.")
        posts = posts[:MAX_CHARS]
    return await summary_chain_async_wrapper({"posts": posts})

async def rag_chain(context: str, user_query: str):
    
    MAX_CHARS = 2400000
    if len(context) > MAX_CHARS:
        context = context[:MAX_CHARS]
    if not context.strip():
        return "I don't have any relevant information to answer this question."
    return await rag_chain_async_wrapper({"context": context, "user_query": user_query})