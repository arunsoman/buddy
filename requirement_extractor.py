import neo4j

# ...

def extract_requirements(text):
    # ...
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

    with driver.session() as session:
        for requirement in requirements:
            # Create a requirement node
            session.run("""
                CREATE (r:Requirement {id: $requirement_id, text: $requirement_text})
            """, requirement_id=requirement_id, requirement_text=requirement_text)

            # Create relationships between requirement nodes
            for related_requirement_id in related_requirement_ids:
                session.run("""
                    MATCH (r1:Requirement {id: $requirement_id})
                    MATCH (r2:Requirement {id: $related_requirement_id})
                    CREATE (r1)-[:CONTAINS {relationship_type: 'Hierarchical'}]->(r2)
                """, requirement_id=requirement_id, related_requirement_id=related_requirement_id)

    driver.close()
