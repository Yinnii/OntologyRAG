import neo4j, os, sys
import numpy as np
import requests
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

def check_parameters_for_equality(run1, run2):
    """
    Check if two runs have similar parameters.
    """
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_SMALL")  # or EMBEDDING_LARGE

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY,
    }

    data1 = {
        "input": str(run1['run']['flow']['hyperparametersettings'])
    }

    data2 = {
        "input": str(run2['run']['flow']['hyperparametersettings'])
    }

    response1 = requests.post(EMBEDDING_ENDPOINT, headers=headers, json=data1)
    response1.raise_for_status()
    embedding1 = response1.json()["data"][0]["embedding"]

    response2 = requests.post(EMBEDDING_ENDPOINT, headers=headers, json=data2)
    response2.raise_for_status()
    embedding2 = response2.json()["data"][0]["embedding"]

    dist = euclidean_distance(embedding1, embedding2)

    if dist < 0.1:
        logger.info("Runs have similar hyperparameters.")
        return True

    return False

def euclidean_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.linalg.norm(vec1 - vec2)

