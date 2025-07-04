# The main file should handle the query with metadata of the dataset
# It connects the retrieving similar data with searching for the best hyperparameters using the MCPClient
# It should be also possible to integrate multiple other MCPClients later

import os, asyncio, sys, uvicorn
from mcp_client.client import MCPClient
from similar_data import find_similar_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ontorag_logger as logger
from fastapi import FastAPI, Request
from graph_creation.store_run import OntologyGraphStoreRun
from graph_creation.connect_postgres import init_postgresql, close_postgresql

app = FastAPI()

OUTPUT_RUN = """
When retrieving runs, extract all the information related to the run and return it in a structured format.
For example:
```{
    "run": {
      "name": "run25673",
      "dataset": {  
        "dataset_name": "iris",
        "qualities": {
          "NumberOfInstances": 150,
          "NumberOfFeatures": 4
        }
      },
      "flow": {
        "implementation": "TreeClassifier",
        "software": "scikit-learn",
        "hyperparametersettings": {
          "max_depth": 3,
          "min_samples_split": 2
        }
      },
      "evaluation": {
        "measure": "predictive_accuracy",
        "value": 0.95
      }
    }
}```
"""

@app.post("/retrieve_parameters")
async def retrieve_parameters(request: Request):
    """Retrieve the best hyperparameters for a dataset based on its description.
    This endpoint receives a dataset description as input, searches for similar datasets in the Neo4j database,
    and retrieves the best hyperparameters for the dataset using the MCPClient.
    """
    req = await request.json()
    query = req.get("query", "")

    logger.info("Start searching for settings for the dataset...")
    client = MCPClient()
    await client.connect_to_server("src/mcp_server/server.py")

    response = ""

    similar_dataset = find_similar_dataset(query)

    if not similar_dataset:
        logger.info("No similar datasets found.")
        # This will not return a good solution
        response = await client.query(f"Based on your knowledge, what are the best hyperparameters for a dataset with the following description: {query}?")
    else:
        logger.info(f"Found similar dataset: {similar_dataset['name']}")
        response = await client.query(f"Retrieve the best hyperparametersettings for the following dataset {similar_dataset['name']}?")

    await client.cleanup()
    logger.info(f"Response: {response}")
    await client.cleanup()
    return {"message": response}

@app.post("/retrieve_runs")
async def retrieve_runs(request: Request):
    """Retrieve runs from the neo4j based on the given metadata and description of a dataset."""
    req = await request.json()
    query = req.get("query", "")
    
    logger.info("Start searching for runs in the graph database...")
    client = MCPClient()
    await client.connect_to_server("src/mcp_server/server.py")

    response = ""
    try:
        similar_dataset = find_similar_dataset(query) 
        if not similar_dataset:
            logger.info("No similar datasets found.")
            response = await client.query_for_run(f"Based on your knowledge, what are the 3 best runs for a dataset with the following description: {query}?")
        else:
            logger.info(f"Found similar dataset: {similar_dataset['name']}")
            response = await client.query_for_run(f"Retrieve the 3 best runs for the following dataset {similar_dataset['name']}." + OUTPUT_RUN)
        
        await client.cleanup()
        return {"message": response}
    except Exception as e:
        logger.error(f"Error retrieving runs: {e}")
        response = f"Error retrieving runs: {str(e)}"
        return {"message": response}

@app.post("/store_run")
async def store_run(request: Request):
    """
    Store a new run in the ontology graph.
    This endpoint receives a run's metadata and stores it in the Neo4j graph database.
    It expects a JSON payload with the run's details.
    """
    URI = os.getenv("NEO4J_URI", "neo4j://localhost:7688")
    USER = os.getenv("NEO4J_USER", "neo4j")
    PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    connection = init_postgresql()

    # initialize the graph
    request_data = await request.json()
    try:
      ontoGraph = OntologyGraphStoreRun()
      ontoGraph.insert_new_run(request_data)

      # store .pkl file that has  into postgresql database with run id
      run_id = request_data.get("run_id")
      # model is a pickle.dumps object as bytes
      model = request_data.get("model")
      with connection.cursor() as cursor:
          cursor.execute(
              "CREATE TABLE IF NOT EXISTS runs (run_id VARCHAR PRIMARY KEY, model BYTEA)"
          )
          cursor.execute(
              "INSERT INTO runs (run_id, model) VALUES (%s, %s) ON CONFLICT (run_id) DO UPDATE SET model = EXCLUDED.model",
              (run_id, model)
          )
          connection.commit()
      
      close_postgresql(connection)

      return {"message": "Run stored successfully."}
    except Exception as e:
      logger.error(f"Error storing run: {e}")
      return {"message": "Error storing run.", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6666)

