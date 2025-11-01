from neo4j import Driver
import logging
import asyncio 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class GraphAnalyzer:
    """
    Class for analyzing graphs using Neo4j Graph Data Science (GDS) library.
    """

    def __init__(self, driver: Driver):
        self.driver = driver

    def drop_graph(self, graph_name: str):
        """
        Remove a GDS graph projection from memory if it exists.
        This is essential to make the script re-runnable.
        """
        query = """
        CALL gds.graph.exists($graph_name) YIELD exists
        WHERE exists
        CALL gds.graph.drop($graph_name) YIELD graphName
        RETURN graphName
        """
        with self.driver.session(database="neo4j") as session:
            result = session.run(query, graph_name=graph_name).single()
            if result:
                logging.info(f"Removed existing GDS graph projection: '{result['graphName']}'")


    def project_alliance_graph(self, graph_name: str, confidence_threshold: float = 0.5):
        """
        Projects the alliance graph into GDS memory using a Cypher projection.
        Uses an undirected pattern to ensure compatibility with the Leiden algorithm.
        """
        query = """
        CALL gds.graph.project(
        $graph_name,
        'User',
        {
            AGREES_WITH: {
            type: 'AGREES_WITH',
            orientation: 'UNDIRECTED',
            properties: {
                weight: {
                property: 'weight',
                defaultValue: 0.0
                }
            }
            }
        }
        )
        YIELD graphName, nodeCount, relationshipCount
        """
        with self.driver.session(database="neo4j") as session:
            result = session.run(query, graph_name=graph_name).single()
            if result:
                logging.info(f"Graph '{result['graphName']}' projected with {result['nodeCount']} nodes and {result['relationshipCount']} relationships.")
            return result

    def run_leiden(self, graph_name: str, community_property: str = "communityId", gamma_value: float = 1.5):
        """
        Runs the Leiden algorithm for community detection in 'mutate' mode.

        :param graph_name: Name of the graph to run the algorithm on.
        :param community_property: Name of the property to save the community ID.
        :param gamma_value: Resolution parameter. Values > 1.0 increase the number of communities.
        """
        query = """
        CALL gds.leiden.write(
            $graph_name,
            {
                writeProperty: $community_property,
                relationshipWeightProperty: 'weight',
                gamma: $gamma_value
            }
        )
        YIELD
            communityCount,
            nodePropertiesWritten
        """
        with self.driver.session(database="neo4j") as session:
            result = session.run(
                query,
                graph_name=graph_name,
                community_property=community_property,
                gamma_value=gamma_value
            ).single()
            if result:
                logging.info(f"Leiden algorithm completed with gamma={gamma_value}. Found {result['communityCount']} communities. Results written to '{community_property}'.")
            return result

    def get_leiden_communities(self, graph_name: str, community_property: str = "communityId"):
        """
        Retrieves the detected communities from the graph.
        Uses gds.graph.nodeProperty.stream to read the node property from the projected graph.

        :param graph_name: Graph name.
        :param community_property: Property that contains the community ID.
        """
        query = f"""
        CALL gds.graph.nodeProperty.stream('{graph_name}', '{community_property}')
        YIELD nodeId, propertyValue AS communityId
        RETURN communityId, COLLECT(gds.util.asNode(nodeId).name) AS members
        ORDER BY size(members) DESC
        """
        with self.driver.session(database="neo4j") as session:
            return [record.data() for record in session.run(query)]

    def close(self):
        """
        Chiude la connessione al database.
        """
        if self.driver:
            self.driver.close()