import neo4j, os
from random import random
from openml import flows
from sentence_transformers import SentenceTransformer
from neo4j_graphrag.embeddings import OllamaEmbeddings

class VectorIndexing:
    """
    This script connects to a Neo4j database, retrieves nodes, generates embeddings using OpenAI's API,
    and stores these embeddings back into the nodes in the database.
    """
    openai_token = None 
    db_name = 'neo4j'
    model = SentenceTransformer('all-MiniLM-L6-v2')
    openai_model = os.getenv("EMBEDDING_MODEL", None)

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

    def create_vector_index(self, node_label='Dataset', embedding_property='embedding', vector_dimensions=384, similarity_function='euclidean'):
            # Create a vector index for the embedding property on Dataset nodes
            self.driver.execute_query('''
            CREATE VECTOR INDEX $index_name IF NOT EXISTS
            FOR (n:$node_label) ON (n.$embedding_property) 
            OPTIONS {indexConfig: {`vector.dimensions`: $vector_dimensions, `vector.similarity_function`: $similarity_function}}
            ''', index_name=node_label+"_"+embedding_property ,node_label=node_label, embedding_property=embedding_property, 
                 vector_dimensions=vector_dimensions, similarity_function=similarity_function
            )
            print(f"Vector index created for {node_label} nodes with embedding property {embedding_property}.")

    def create_embeddings(self, node_label='Dataset', embedding_property='embedding'):
        """
        Generate embeddings for the description of the dataset nodes in the database in batches.
        """
        batch_size = 100
        batch_n = 1
        node_batch = []

        with self.driver.session(database=self.db_name) as session:
            try: 
              result = session.run(f'''MATCH (n:{node_label} WHERE n.description IS NOT null RETURN n.name AS name, n.description AS description''') 
            except Exception as e:
              result = session.run(f'''MATCH (n:{node_label}) WHERE n.hasValue IS NOT null RETURN n.name AS name, n.hasValue AS description''')

            for record in result:
                name = record.get('name')
                description = record.get('description', None)

                if name is not None:
                    if self.openai_token is None:
                        node_batch.append({
                            'name': name,
                            'description': description,
                            'embedding': self.model.encode(f'''
                                        Name: {name}, Description: {description}'''),
                        })
                    else:
                      node_batch.append({
                          'name': name,
                          'description': description,
                          'to_encode': f'Name: {name} Description: {description}'
                      })

                # Import a batch; flush buffer
                if len(node_batch) == batch_size:
                    self.import_batch(node_batch, embedding_property, batch_n)
                    node_batch = []
                    batch_n += 1

            # Flush last batch
            self.import_batch(node_batch, embedding_property, batch_n)

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

    def import_batch(self, nodes, embedding_property, batch_n):

      if self.openai_token is None:
          self.driver.execute_query('''
          UNWIND $nodes AS node    
          MATCH (n {name: node.name})
          CALL db.create.setNodeVectorProperty(n, $embedding_property, node.embedding)
          ''', nodes=nodes, database_=self.db_name, embedding_property=embedding_property)
          print(f'Processed batch {batch_n}.')

      else:
        # Generate and store embeddings for nodes
        self.driver.execute_query('''
        CALL genai.vector.encodeBatch($listToEncode, 'AzureOpenAI', { token: $token, resource: $resource, deployment: 'text-embedding-3-small'}) YIELD index, vector
        MATCH (n {name: $nodes[index].name})
        CALL db.create.setNodeVectorProperty(n, $embedding_property, vector)
        ''', nodes=nodes, listToEncode=[node['to_encode'] for node in nodes], token=self.openai_token, resource='gpt-dbis', embedding_property=embedding_property,
        database_=self.db_name)
        print(f'Processed batch {batch_n}')


if __name__ == "__main__":
    VectorIndexing(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=("neo4j", os.getenv("NEO4J_PASSWORD", "password")),
        db_name="neo4j",
        openai_token=os.getenv("OPENAI_API_KEY", None)).create_embeddings()