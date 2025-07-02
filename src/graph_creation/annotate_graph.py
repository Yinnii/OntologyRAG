from openml import tasks, runs, evaluations
import warnings
import os
import pandas as pd
import psycopg2 as p
from graph_creation.create_graph import OntologyGraph
from graph_creation.vector_index import VectorIndexing
from sqlalchemy import create_engine
import asyncio
warnings.filterwarnings("ignore", category=RuntimeWarning)

POSTGRES_USER = os.getenv("POSTGRES_USER", "openml_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "openml_password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "openml")

# initialize postgresql database connection
def init_postgresql():
    """ Initialize the PostgreSQL database connection."""
    connection = p.connect(
      database=POSTGRES_DB,
      user=POSTGRES_USER,
      password=POSTGRES_PASSWORD,
      host=POSTGRES_HOST,
      port=POSTGRES_PORT
    )

    engine = create_engine(
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    return connection, engine

def close_postgresql(connection):
    """ Close the PostgreSQL database connection."""
    if connection:
        connection.close()
        print("PostgreSQL connection is closed.")
    else:
        print("No PostgreSQL connection to close.")

def insert_runs_into_graph():
    """ Insert OpenML runs into the Neo4j graph database.
    This function retrieves all runs from OpenML tasks of type SUPERVISED_CLASSIFICATION
    and inserts them into a Neo4j graph database using the OntologyGraph class.
    """
    URI = os.getenv("NEO4J_URI", "neo4j://localhost:7688")
    USER = os.getenv("NEO4J_USER", "neo4j")
    PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    # TOKEN = os.getenv("OPENAI_API_KEY", None)

    connection, engine = init_postgresql()

    # initialize the graph
    graph = OntologyGraph(URI, user=USER, password=PASSWORD)
    # load ontology only if one initializes a new graph
    graph.load_ontology()

    task_list = tasks.list_tasks(task_type=tasks.TaskType.SUPERVISED_CLASSIFICATION, output_format='dataframe')
    task_list['ttid'] = task_list['ttid'].astype(str)
    task_list.to_sql('tasks', con=engine, if_exists='replace', index=False)
    print(f"Number of tasks: {len(task_list)}")
    
    data_eval = [12, 31, 1067, 41143, 41162, 42733, 4538, 40498, 40984]
    task_list = task_list[task_list['did'].isin(data_eval)]

    # get run_ids from postgresql for each task
    for i, task in task_list.iterrows():
      tid = task['tid']
      
      eval_for_task = get_evaluations_from_task(tid, engine=engine)

      if eval_for_task.empty:
          print(f"No evaluations found for task {tid}. Skipping.")
          continue

      for i, eval in eval_for_task.iterrows():
        run_id = eval['run_id']
        print(f"Processing run {run_id} for task {tid}...")

        # check if run already exists in the graph
        if graph.run_exists(run_id):
            print(f"Run {run_id} already exists in the graph. Skipping.")
            continue

        # insert run into the graph
        try:
          graph.insert_run(run_id=run_id)
        except Exception as e:
          print(f"Error inserting run {run_id} into the graph: {e}")
          continue
      
        # TODO if the run has an executable model, transform the model into binary and insert it into postgresql with run_id
        # OR execute the run and train the model, then insert the model into postgresql with run_id

    graph.close()
    print("All runs have been inserted into the graph.")

    # create vector index 
    vectorIndexing = VectorIndexing(
        uri=URI,
        auth=(USER, PASSWORD),
        db_name='neo4j',
        openai_token=os.getenv("AZURE_API_KEY", None)  # or None if not using Azure OpenAI 
    )

    # the embeddings are created for the descriptions of hyperparametersettings and datasets
    vectorIndexing.create_embeddings()

    close_postgresql(connection)

def get_evaluations_from_task(task_id, engine=None) -> pd.DataFrame:
    metric = "predictive_accuracy"
    evals = evaluations.list_evaluations(
        function=metric, tasks=[task_id], output_format="dataframe"
    )

    if evals.empty:
        print(f"No evaluations found for task {task_id} with metric {metric}.")
        return pd.DataFrame()
    
    # Sorting the evaluations in decreasing order of the metric chosen
    evals = evals.sort_values(by="value", ascending=False)
    evals.to_sql(f'evaluations_task_{task_id}', con=engine, if_exists='replace', index=False)
    print(f"Evaluations for task {task_id} saved to database.")

    evals = evals.head(100)

    return evals


if __name__ == "__main__":
    insert_runs_into_graph()
