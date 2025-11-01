import logging
from typing import List, Dict, Any

import neo4j

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_all_entities(driver: neo4j.Driver) -> List[str]:
    query = "MATCH (e:PoliticalEntity) RETURN e.name AS name ORDER BY name"
    try:
        with driver.session(database="neo4j") as session:
            result = session.run(query)
            return [record["name"] for record in result]
    except Exception as e:
        logging.error(f"Errore in get_all_entities: {e}")
        return []

def get_entity_overview_data(driver: neo4j.Driver, entity_name: str) -> Dict[str, Any]:
    stance_query = """
        MATCH (e:PoliticalEntity {name: $entity_name})<-[r:HAS_STANCE]-(p:Post)
        RETURN r.stance AS stance, count(p) AS count
    """
    graph_query = """
        MATCH (u:User)-[:POSTED]->(p:Post)-[:MENTIONS]->(e:PoliticalEntity {name: $entity_name})
        RETURN u.name AS source, e.name AS target LIMIT 25
    """
    overview_data = {"stance_distribution": [], "graph_data": {"nodes": [], "edges": []}}
    try:
        with driver.session(database="neo4j") as session:
            stance_result = session.run(stance_query, entity_name=entity_name)
            overview_data["stance_distribution"] = [{"stance": r["stance"], "count": r["count"]} for r in stance_result]
            
            graph_result = session.run(graph_query, entity_name=entity_name)
            nodes, edges = set(), []
            for record in graph_result:
                nodes.add(record["source"])
                nodes.add(record["target"])
                edges.append({"source": record["source"], "target": record["target"]})
            overview_data["graph_data"]["nodes"] = list(nodes)
            overview_data["graph_data"]["edges"] = edges
        return overview_data
    except Exception as e:
        logging.error(f"Errore in get_entity_overview_data: {e}")
        return overview_data