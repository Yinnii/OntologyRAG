import neo4j, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ontorag_logger as logger

URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
AUTH = (os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'password'))
DB_NAME = 'neo4j'
driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()

openAI_token = os.getenv("AZURE_API_KEY", None)

search_query = '''
WITH genai.vector.encode($searchPrompt, 'AzureOpenAI', { token: $token, resource: $resource, deployment: 'text-embedding-3-small'}) AS queryVector
CALL db.index.vector.queryNodes('embedding', 5, queryVector)
YIELD node, score
RETURN node.name as name, node.description, score
'''

def find_similar_dataset(search_prompt: str):
    records, _, _ = driver.execute_query(
        search_query, searchPrompt=search_prompt, token=openAI_token, resource='gpt-dbis',
        database_=DB_NAME)

    if not records:
        logger.info("No similar datasets found.")
        return None
    logger.info(f"Found {len(records)} similar datasets.")
    records.sort(key=lambda x: x['score'], reverse=True)

    return records[0]