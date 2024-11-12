import networkx as nx
import neo4j

def construct_knowledge_graph(entities, relationships):
    # Create a new directed graph
    G = nx.DiGraph()
    
    # Add nodes for entities
    for entity in entities:
        G.add_node(entity[0], type=entity[1])
    
    # Add edges for relationships
    for relationship in relationships:
        G.add_edge(relationship[0], relationship[1], type=relationship[2])
    
    # Optionally, store the graph in Neo4j
    # driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    # with driver.session() as session:
    #     session.write_transaction(lambda tx: tx.run("CREATE (n:Entity {name:$name, type:$type})", name=entity[0], type=entity[1]))
    
    return G

    import networkx as nx
import neo4j

def construct_knowledge_graph(entities, relationships):
    # Create a new directed graph
    G = nx.DiGraph()
    
    # Add nodes for entities
    for entity in entities:
        G.add_node(entity[0], type=entity[1])
    
    # Add edges for relationships
    for relationship in relationships:
        G.add_edge(relationship[0], relationship[1], type=relationship[2])
    
    # Store the graph in Neo4j (optional)
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    with driver.session() as session:
        # Clear the graph (if it already exists)
        session.run("MATCH (n) DETACH DELETE n")
        
        # Create nodes for entities
        for entity in entities:
            session.run("CREATE (n:Entity {name:$name, type:$type})", name=entity[0], type=entity[1])
        
        # Create edges for relationships
        for relationship in relationships:
            session.run("""
                MATCH (a:Entity {name:$source})
                MATCH (b:Entity {name:$target})
                CREATE (a)-[:RELATIONSHIP {type:$type}]->(b)
            """, source=relationship[0], target=relationship[1], type=relationship[2])
    
    driver.close()
    return G
