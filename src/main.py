# The main file should handle the query with metadata of the dataset
# It connects the retrieving similar data with searching for the best hyperparameters using the MCPClient
# It should be also possible to integrate multiple other MCPClients later

import os, sys, uvicorn, uuid, json, re, json_repair, datetime
from .mcp_client.client import MCPClient
from .mcp_client.store_client import MCPStoreClient
from similar_data import find_similar_dataset, check_parameters_for_equality
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
      "name": "25673",
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
def _store_tokens(tokens):
    try:
      logger.info(tokens)
      connection, _ = init_postgresql()
      # initialize a uuid for the query
      query_id = str(uuid.uuid4())
      date = datetime.datetime.now()

      with connection.cursor() as cursor:
        cursor.execute("CREATE TABLE IF NOT EXISTS openai_tokens (date TIMESTAMP DEFAULT CURRENT_TIMESTAMP, query_id VARCHAR PRIMARY KEY, completion_tokens INT, prompt_tokens INT, total_tokens INT, embedding_tokens INT)")
        completion_tokens = int(tokens.get("completion_tokens") or 0)
        prompt_tokens = int(tokens.get("prompt_tokens") or 0)
        total_tokens = int(tokens.get("total_tokens") or 0)
        embedding_tokens = int(tokens.get("embedding_tokens") or 0)
        cursor.execute(
            "INSERT INTO openai_tokens (date, query_id, completion_tokens, prompt_tokens, total_tokens, embedding_tokens) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (query_id) DO UPDATE SET completion_tokens = EXCLUDED.completion_tokens, prompt_tokens = EXCLUDED.prompt_tokens, total_tokens = EXCLUDED.total_tokens, embedding_tokens = EXCLUDED.embedding_tokens",
            (date, query_id , completion_tokens, prompt_tokens, total_tokens, embedding_tokens)
        )
        connection.commit()
    
      close_postgresql(connection)

    except Exception as e:
        logger.error(f"Error storing tokens in PostgreSQL: {e}")
        
    return query_id

def _parse_response(response):
    try:
        match = re.findall(r'```json(.*?)```', response, re.DOTALL)
        if not match:
            logger.error(f"No JSON block found in response: {response}")
            return json_objects
        logger.info(f'Response found {match} with {type(match)}')
        json_objects = []
        if len(match) == 1:
            logger.info(f'Only one object found {match[0]} with {type(match[0])}')
            m = json.loads(match[0])
            if isinstance(m, list):
                json_objects = m
            else:
                json_objects.append(m)
            return json_objects
        else:
          for m in match:
              m = json.loads(m)
              try: 
                  if isinstance(m, list):
                     logger.info(f'Append object to json_objects {m[0]} with {type(m[0])}')
                     json_objects.extend(m)
                  else:
                      json_objects.append(m)
                  return json_objects
              except Exception as e:
                  logger.error(f"JSON decode error for block: {m}\nError: {e}")
                  # Try to repair JSON
                  try:
                      json_objects.append(json_repair.loads(m.strip()))
                      return json_objects
                  except Exception as e2:
                      logger.error(f"json_repair failed for block: {m}\nError: {e2}")
                      continue
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        return json_objects

def _check_response_distinctness(response):
    logger.info(f'Response is {response} with type {type(response)}')
    if isinstance(response, list) and len(response) > 1:
        for i in range(len(response) - 1):
            for j in range(i + 1, len(response)):
                equal, tokens = check_parameters_for_equality(response[i], response[j])
                if equal:
                    logger.warning(f"Responses {i} and {j} have the same hyperparametersettings: {response[i]['run']['flow']['hyperparametersettings']}")
                    return False, tokens
        return True, tokens
    elif isinstance(response, dict):
        runs = response.get("runs", [])
        if len(runs) > 1:
            for i in range(len(runs) - 1):
                for j in range(i + 1, len(runs)):
                    equal, tokens = check_parameters_for_equality(runs[i], runs[j])
                    if equal:
                        logger.warning(f"Responses {i} and {j} have the same hyperparametersettings: {runs[i]['run']['flow']['hyperparametersettings']}")
                        return False, tokens
            return True, tokens
        return True, 0
    else:
        logger.error("Unexpected response format.")
        # Since we do not want to disturb the flow, we assume the response is distinct
        return True, 0

# @app.post("/retrieve_parameters")
# async def retrieve_parameters(request: Request):
#     """Retrieve the best hyperparameters for a dataset based on its description.
#     This endpoint receives a dataset description as input, searches for similar datasets in the Neo4j database,
#     and retrieves the best hyperparameters for the dataset using the MCPClient.
#     """
#     req = await request.json()
#     query = req.get("query", "")

#     logger.info("Start searching for settings for the dataset...")
#     client = MCPClient()
#     await client.connect_to_server("./mcp_server/server.py")

#     response = ""

#     similar_dataset = find_similar_dataset(query)

#     if not similar_dataset:
#         logger.info("No similar datasets found.")
#         # This will not return a good solution
#         response, tokens = await client.query(f"Based on your knowledge, what are the best hyperparameters for a dataset with the following description: {query}?")
#     else:
#         logger.info(f"Found similar dataset: {similar_dataset['name']}")
#         response, tokens = await client.query(f"Retrieve the best hyperparametersettings for the following dataset {similar_dataset['name']}?")

#     query_id = _store_tokens(tokens)
#     logger.info(f"Tokens stored with query_id: {query_id}")

#     logger.info(f"Response: {response}")
#     await client.cleanup()
#     return {"message": response}

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
        similar_dataset, embedding_tokens = find_similar_dataset(query) 
        if not similar_dataset:
            logger.info("No similar datasets found.")
            response, tokens = await client.query_for_run(f"Based on your knowledge, what are the 3 best runs for a dataset with the following description: {query}?")
        else:
            logger.info(f"Found similar dataset: {similar_dataset['name']}")
            response, tokens = await client.query_for_run(f"Retrieve the 3 best runs for the following dataset {similar_dataset['name']}." + OUTPUT_RUN)

        await client.cleanup()

        response = _parse_response(response)

        tokens["embedding_tokens"] += sum(embedding_tokens) if isinstance(embedding_tokens, list) else embedding_tokens

        is_distinct, embedding_tokens = _check_response_distinctness(response)
        tokens["embedding_tokens"] += sum(embedding_tokens) if isinstance(embedding_tokens, list) else embedding_tokens

        if not is_distinct:
            logger.warning("Response contains runs with the same hyperparametersettings, removing them.")
            
            if isinstance(response, list):
                dist_num = len(response) - 1
                response = {"runs": [response[0]]}
            elif isinstance(response, dict):
                dist_num = len(response.get("runs")) - 1
                response = {"runs": [response.get("runs")[0]]}

            # find a random run with different hyperparametersettings
            await client.connect_to_server("src/mcp_server/server.py")
            response_new, tokens_new = await client.query_for_run(f"Retrieve {dist_num} random runs for the following dataset {similar_dataset['name']}. Use only one json in the response." + OUTPUT_RUN)
            await client.cleanup()

            response_new = _parse_response(response_new)

            if isinstance(response_new, list):
                response["runs"].extend(response_new)
            elif isinstance(response_new, dict):
                try:
                    response["runs"].extend(response_new["runs"])
                except Exception as e:
                    logger.error(f"Error extending runs: {e}")
                    logger.error(f"Response new: {response_new}")
                    return {"message": response}
                    

            tokens["completion_tokens"] += tokens_new["completion_tokens"]
            tokens["prompt_tokens"] += tokens_new["prompt_tokens"]
            tokens["total_tokens"] += tokens_new["total_tokens"]

        query_id = _store_tokens(tokens)
        logger.info(f"Tokens stored with query_id: {query_id}")

        return {"message": response}
    except Exception as e:
        logger.error(f"Error retrieving runs: {e}")
        response = f"Error retrieving runs: {str(e)}"
        return {"message": response}

@app.post("/store_run_mcp")
async def store_run(request: Request):
    """
    Store a new run using the JSON payload using MCP client/server.
    """
    run_details = await request.json()
    client = MCPStoreClient()
    await client.connect_to_server("src/mcp_server/write_server.py")
    response = await client.store_run(run_details)
    await client.cleanup()
    return {"message": response}

@app.post("/store_run_static")
async def store_run_static(request: Request):
    """
    Store a new run in the ontology graph.
    This endpoint receives a run's metadata and stores it in the Neo4j graph database.
    It expects a JSON payload with the run's details.
    """

    connection, _ = init_postgresql()

    run_id = str(uuid.uuid4())
    request_data = await request.json()

    try:
      # ontoGraph = OntologyGraphStoreRun()
      # run_id = ontoGraph.insert_new_run(run)
      # logger.info(f"Run {run_id} stored successfully in the graph database.")
      # model is a pickle.dumps object as bytes
      run = json.loads(request_data.get("run"))

      with connection.cursor() as cursor:
          cursor.execute(
              "CREATE TABLE IF NOT EXISTS runs (run_id VARCHAR PRIMARY KEY, details TEXT)"
          )
          cursor.execute(
              "INSERT INTO runs (run_id, details) VALUES (%s, %s) ON CONFLICT (run_id) DO UPDATE SET details = EXCLUDED.details",
              (run_id, json.dumps(run))
          )
          connection.commit()
      
      close_postgresql(connection)
      logger.info(f"Run {run_id} stored successfully in PostgreSQL.")

      return {"success": True}
    except json.JSONDecodeError as e:
      logger.warning(f"JSON decode error: {e}")

      with connection.cursor() as cursor:
          cursor.execute(
              "CREATE TABLE IF NOT EXISTS runs (run_id VARCHAR PRIMARY KEY, details TEXT)"
          )
          cursor.execute(
              "INSERT INTO runs (run_id, details) VALUES (%s, %s) ON CONFLICT (run_id) DO UPDATE SET details = EXCLUDED.details",
              (run_id, str(request_data))
          )
          connection.commit()
      return {"success": False, "error": "Invalid JSON format."}
    except Exception as e:
      logger.error(f"Error storing run: {e}")
      return {"success": False, "error": str(e)}


