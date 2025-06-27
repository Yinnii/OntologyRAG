# The main file should handle the query with metadata of the dataset
# It connects the retrieving similar data with searching for the best hyperparameters using the MCPClient
# It should be also possible to integrate multiple other MCPClients later

import os, asyncio, sys, uvicorn
from mcp_client.client import MCPClient
from similar_data import find_similar_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ontorag_logger as logger
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/retrieve_parameters")
async def retrieve_parameters(request: Request):
    req = await request.json()
    query = req.get("query", "")

    logger.info("Start searching for settings for the dataset...")
    client = MCPClient()
    await client.connect_to_server("src/mcp_server/mcp-neo4j-cypher/src/mcp_neo4j_cypher/server.py")

    response = ""

    similar_dataset = find_similar_dataset(query)

    if not similar_dataset:
        logger.info("No similar datasets found.")
        # This will not return a good solution
        response = await client.query(f"Based on your knowledge, what are the best hyperparameters for a dataset with the following description: {query}?")
    else:
        logger.info(f"Found similar dataset: {similar_dataset['name']}")
        response = await client.query(f"Retrieve the best hyperparametersettings for the following dataset {similar_dataset['name']}?")

    logger.info(f"Response: {response}")
    await client.cleanup()
    return {"message": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6666)

