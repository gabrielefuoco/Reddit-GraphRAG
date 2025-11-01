from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from neo4j import Driver

from src.llm.core import llm_text
from src.pipeline.rag_chain import GraphRAGPipeline


AGENT_PROMPT_TEMPLATE = """You are a reasoning agent. Your mission is to answer the user's question using the provided tools. You must follow the format below precisely.

**TOOLS:**
{tools}

**CONVERSATION HISTORY:**
{chat_history}

**RESPONSE FORMAT:**
You must output a single block of text following this exact structure. Do not deviate.

Thought: Your brief analysis of the user's question and your plan. Decide which tool to use.
Action: The single tool name from [{tool_names}].
Action Input: The input for the tool.
Observation: The result from the tool.
Thought: I will now decide if I need another tool or if I can give the final answer.
Final Answer: The comprehensive answer to the user's original question.

---
### CRITICAL RULE ###
After receiving an Observation, you have only TWO choices:
1.  Start a new Thought/Action/Action Input cycle if you NEED more information.
2.  Provide the 'Final Answer:' if the Observation contains everything required to answer the user's question.
If the information is sufficient, you MUST provide the 'Final Answer:'. Do not get stuck in a loop.
---

Begin execution now.

User Question: {input}
Thought:{agent_scratchpad}"""


def create_political_agent(driver: Driver) -> AgentExecutor:
    """
    Factory per creare l'agente ReAct con i suoi strumenti.
    """
    rag_pipeline = GraphRAGPipeline(driver=driver)

    political_analyzer_tool = Tool(
        name="political_analyzer",
        func=rag_pipeline.query,
        description=(
            "Use this tool to answer questions about political opinions, criticism, "
            "support or analysis regarding political figures, organizations or concepts. "
            "Input should be the complete user question."
        ),
    )
    tools = [political_analyzer_tool]

    prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    agent = create_react_agent(
        llm=llm_text,
        tools=tools,
        prompt=prompt,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        # True per dare al parser una chance di correggere l'output dell'LLM.
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=False,
    )