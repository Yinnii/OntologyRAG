import litellm
import neo4j, os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ontorag_logger as logger

URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
AUTH = (os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'password'))
DB_NAME = 'neo4j'
driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()

search_query = f'''
CALL db.index.vector.queryNodes('embedding', 5, $queryVector)
YIELD node, score
RETURN node.name as name, node.description, score
'''

def find_similar_dataset(search_prompt: str):

    response = litellm.embedding(
        api_base=os.getenv("OPENAI_ENDPOINT"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        dimensions=1536,
        input=search_prompt,
    )

    records, _, _ = driver.execute_query(
        search_query, queryVector=response['data'][0]['embedding'],
        database_=DB_NAME)

    if not records:
        logger.info("No similar datasets found.")
        return None
    logger.info(f"Found {len(records)} similar datasets.")
    records.sort(key=lambda x: x['score'], reverse=True)

    tokens = response['usage']['total_tokens']
    return records[0], tokens

def check_parameters_for_equality(run1, run2):
    """
    Check if two runs have similar parameters.
    """
    total_token = 0

    response1 = litellm.embedding(
        api_base=os.getenv("OPENAI_ENDPOINT"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        dimensions=1536,
        input=str(run1['run']['flow']['hyperparametersettings']),
    )
    total_token += response1['usage']['total_tokens']

    response2 = litellm.embedding(
        api_base=os.getenv("OPENAI_ENDPOINT"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        dimensions=1536,
        input=str(run2['run']['flow']['hyperparametersettings']),
    )
    total_token += response2['usage']['total_tokens']

    dist = euclidean_distance(response1['data'][0]['embedding'], response2['data'][0]['embedding'])

    if dist < 0.1:
        logger.info("Runs have similar hyperparameters.")
        return True, total_token

    return False, total_token

def euclidean_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.linalg.norm(vec1 - vec2)

