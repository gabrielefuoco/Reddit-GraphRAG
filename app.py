import streamlit as st
import pandas as pd
import plotly.express as px
from neo4j import GraphDatabase
from streamlit_agraph import agraph, Node, Edge, Config
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.react_agent import create_political_agent
from src.utils.config import load_credentials
from src.pipeline.query_templates import get_all_entities, get_entity_overview_data

st.set_page_config(
    page_title="Reddit Political Opinion Analyzer (GraphRAG)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Reddit Political Opinion Analyzer ")
st.caption("Interfaccia per interrogare il Knowledge Graph politico costruito da Reddit usando Neo4j e LangChain")

@st.cache_resource
def init_connections():
    """
    Inizializza il driver Neo4j e l'agente.
    """
    creds = load_credentials()
    uri, user, password = creds.get("neo4j_uri"), creds.get("neo4j_user"), creds.get("neo4j_password")
    if not all([uri, user, password]):
        st.error("Credenziali Neo4j mancanti. Configura il .env.")
        st.stop()
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        agent_executor = create_political_agent(driver)
        return driver, agent_executor
    except Exception as e:
        st.error(f"Errore di connessione a Neo4j: {e}")
        st.stop()

driver, agent_executor = init_connections()

st.sidebar.title("Viste")
selected_view = st.sidebar.selectbox("Seleziona una vista", ("Chat", "Panoramica Entità"))

# Inizializza la cronologia della chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ciao, come posso aiutarti?"}]


if selected_view == "Chat":
    st.subheader("Conversazione")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Domanda..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Sto pensando..."):
                try:
                    # Formatta la cronologia per LangChain
                    history = []
                    for msg in st.session_state.messages[:-1]: # Escludi l'input corrente
                        if msg["role"] == "user":
                            history.append(HumanMessage(content=msg["content"]))
                        else:
                            history.append(AIMessage(content=msg["content"]))
                    
                    # Invocazione con cronologia @@@@@@
                    response = agent_executor.invoke({
                        "input": prompt,
                        "chat_history": history
                    })
                    
                    answer = response.get("output", "Qualcosa è andato storto.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Errore: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# --- PANORAMICA ENTITÀ ---

elif selected_view == "Panoramica Entità":
    st.subheader("Panoramica delle Entità Politiche")
    with st.spinner("Caricamento entità..."):
        entities = get_all_entities(driver)
    if not entities:
        st.warning("Nessuna entità politica trovata nel database.")
    else:
        selected_entity = st.selectbox("Seleziona un'entità da analizzare", options=entities)
        if selected_entity:
            with st.spinner(f"Caricamento dati per {selected_entity}..."):
                data = get_entity_overview_data(driver, selected_entity)
            st.markdown(f"### Dashboard per: **{selected_entity}**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Distribuzione delle Stance")
                stance_data = data.get("stance_distribution", [])
                if stance_data:
                    df_stance = pd.DataFrame(stance_data).set_index('stance')
                    st.bar_chart(df_stance)
                else:
                    st.info("Nessun dato sulla stance disponibile.")
            with col2:
                st.markdown("#### Grafo delle Menzioni (Utenti -> Entità)")
                graph_data = data.get("graph_data", {})
                nodes_data, edges_data = graph_data.get("nodes", []), graph_data.get("edges", [])
                if nodes_data and edges_data:
                    nodes = [Node(id=n, label=n, size=15) for n in nodes_data]
                    edges = [Edge(source=e['source'], target=e['target'], type="CURVE_SMOOTH") for e in edges_data]
                    agraph_config = Config(width=500, height=500, directed=True, nodeHighlightBehavior=True)
                    agraph(nodes=nodes, edges=edges, config=agraph_config)
                else:
                    st.info("Nessun dato sul grafo disponibile.")

