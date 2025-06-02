import neo4j, os
from sentence_transformers import SentenceTransformer

class VectorIndexing:
    """
    This script connects to a Neo4j database, retrieves nodes, generates embeddings using OpenAI's API,
    and stores these embeddings back into the nodes in the database.
    """
    openai_token = None 
    db_name = 'neo4j'
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def __init__(self, uri, auth, db_name, openai_token=None):
        """
        Initialize the VectorIndexWithGPT class.
        :param uri: URI for the Neo4j database.
        :param auth: Authentication tuple for the Neo4j database (username, password).
        :param db_name: Name of the database to connect to.
        :param openai_token: OpenAI API token for generating embeddings.
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=auth)
        self.driver.verify_connectivity()
        self.db_name = db_name
        self.openai_token = openai_token

    def create_embeddings(self):
        """
        Generate embeddings for all nodes in the database in batches.
        """
        batch_size = 100
        batch_n = 1
        node_batch = []

        with self.driver.session(database=self.db_name) as session:
            # Fetch all nodes 
            result = session.run('MATCH (n) WHERE n.id IS NOT null RETURN n.id AS name')
            for record in result:
                name = record.get('name')

                if name is not None:
                    if self.openai_token is None:
                        node_batch.append({
                            'name': name,
                            'embedding': self.model.encode(f'''
                                        Name: {name}'''),
                        })
                    else:
                      node_batch.append({
                          'name': name,
                          'to_encode': f'Name: {name}'
                      })

                # Import a batch; flush buffer
                if len(node_batch) == batch_size:
                    self.import_batch(node_batch, batch_n)
                    node_batch = []
                    batch_n += 1

            # Flush last batch
            self.import_batch(node_batch, batch_n)

        # Import complete, show counters
        records, _, _ = self.driver.execute_query('''
        MATCH (n) WHERE n.embedding IS NOT NULL
        RETURN count(*) AS countNodesWithEmbeddings, size(n.embedding) AS embeddingSize
        ''', database_=self.db_name)
        print(f"""
              Embeddings generated and attached to nodes.
              Nodes with embeddings: {records[0].get('countNodesWithEmbeddings')}.
              Embedding size: {records[0].get('embeddingSize')}.
                  """)

    def import_batch(self, nodes, batch_n):

      if self.openai_token is None:
          self.driver.execute_query('''
          UNWIND $nodes AS node    
          MATCH (n {id: node.name})
          CALL db.create.setNodeVectorProperty(n, 'embedding', node.embedding)
          ''', nodes=nodes, database_=self.db_name)
          print(f'Processed batch {batch_n}.')

      else:
        # Generate and store embeddings for nodes
        self.driver.execute_query('''
        CALL genai.vector.encodeBatch($listToEncode, 'OpenAI', { token: $token }) YIELD index, vector
        MATCH (n {id: $nodes[index].name})
        CALL db.create.setNodeVectorProperty(n, 'embedding', vector)
        ''', nodes=nodes, listToEncode=[node['to_encode'] for node in nodes], token=self.openai_token,
        database_=self.db_name)
        print(f'Processed batch {batch_n}')

# if __name__ == "__main__":
#     vectorIndexing = VectorIndexing(
#         uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
#         auth=('neo4j', os.getenv("NEO4J_PASSWORD", "pasword")),
#         db_name='neo4j',
#         openai_token=os.getenv("OPENAI_API_KEY", None)  # Optional, if you want to use OpenAI embeddings
#     )
#     vectorIndexing.create_embeddings()