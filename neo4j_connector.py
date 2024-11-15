import neo4j
from neo4j import GraphDatabase
import neo4j


def populate_neo4j_graph(requirements, classifications, ambiguities):
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

    with driver.session() as session:
        # Populate the Neo4j graph with extracted requirements
        for requirement in requirements:
            session.run("""
                CREATE (r:Requirement {id: $requirement_id, text: $requirement_text})
            """, requirement_id=requirement_id, requirement_text=requirement_text)

        # Populate the Neo4j graph with classifications
        for classification in classifications:
            session.run("""
                MATCH (r:Requirement {id: $requirement_id})
                SET r.classification = $classification
            """, requirement_id=requirement_id, classification=classification)

        # Populate the Neo4j graph with ambiguity detections
        for ambiguity in ambiguities:
            session.run("""
                MATCH (r:Requirement {id: $requirement_id})
                SET r.ambiguity = $ambiguity
            """, requirement_id=requirement_id, ambiguity=ambiguity)

    driver.close()

# Establish a Neo4j connection
def connect_to_neo4j(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver

# Create nodes and relationships
def create_nodes_and_relationships(driver, requirements_dict):
    with driver.session() as session:
        for requirement_id, requirement_data in requirements_dict.items():
            # Create a requirement node
            session.run("""
                CREATE (r:Requirement {id: $requirement_id, description: $description, priority: $priority})
            """, requirement_id=requirement_id, description=requirement_data['description'], priority=requirement_data['priority'])

            # Create attribute nodes and relationships
            for attribute_name, attribute_value in requirement_data['attributes'].items():
                session.run("""
                    CREATE (a:Attribute {name: $attribute_name, value: $attribute_value})
                """, attribute_name=attribute_name, attribute_value=attribute_value)
                session.run("""
                    MATCH (r:Requirement {id: $requirement_id})
                    CREATE (r)-[:HAS_ATTRIBUTE {attribute_name: $attribute_name}]->(a)
                """, requirement_id=requirement_id, attribute_name=attribute_name)

            # Create entity nodes and relationships
            for entity_name, entity_role in requirement_data['entities'].items():
                session.run("""
                    CREATE (e:Entity {name: $entity_name})
                """, entity_name=entity_name)
                session.run("""
                    MATCH (r:Requirement {id: $requirement_id})
                    CREATE (r)-[:INVOLVES_ENTITY {entity_role: $entity_role}]->(e)
                """, requirement_id=requirement_id, entity_role=entity_role)

            # Create relationships between requirements
            for related_requirement_id, relationship_type in requirement_data['relationships'].items():
                session.run("""
                    MATCH (r1:Requirement {id: $requirement_id})
                    MATCH (r2:Requirement {id: $related_requirement_id})
                    CREATE (r1)-[:$relationship_type {relationship_type: $relationship_type}]->(r2)
                """, requirement_id=requirement_id, related_requirement_id=related_requirement_id, relationship_type=relationship_type)

# Close the Neo4j session and driver
session.close()
driver.close()
