from openml import tasks, runs
import warnings
import os
import psycopg2 as p
from create_graph import OntologyGraph
from vector_index import VectorIndexing

warnings.filterwarnings("ignore", category=RuntimeWarning)

# initialize postgresql database connection
def init_postgresql():
    """ Initialize the PostgreSQL database connection."""
    connection = p.connect(
      database="openml",
      user=os.getenv("POSTGRES_USER", "openml_user"),
      password=os.getenv("POSTGRES_PASSWORD", "openml_password"),
      host=os.getenv("POSTGRES_HOST", "localhost"),
      port=os.getenv("POSTGRES_PORT", "5432")
    )

    return connection

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
    TOKEN = os.getenv("OPENAI_API_KEY", None)

    connection = init_postgresql()

    # initialize the graph
    graph = OntologyGraph(URI, user=USER, password=PASSWORD)
    graph.load_ontology()
    # dataset_list = [31, 1067, 41143, 41162, 42733, 12, 4538, 40498, 40984]

    # for dataset_id in dataset_list:
    #   print(f"Dataset ID: {dataset_id}")

    task_list = tasks.list_tasks(task_type=tasks.TaskType.SUPERVISED_CLASSIFICATION, output_format='dataframe')
    print(f"Number of tasks: {len(task_list)}")

    for i, task in task_list.iterrows():
      runs_list = runs.list_runs(task=[task['tid']], output_format='dataframe')
      print(f"Number of runs: {len(runs_list)} for task {task['tid']}")

      with connection.cursor() as cursor:
         cursor.execute('''CREATE TABLE IF NOT EXISTS tasks (tid INTEGER PRIMARY KEY, runs_count INTEGER)''')
         cursor.execute('''INSERT INTO tasks VALUES (%s, %s)''', (task['tid'], len(runs_list)))

      for j, run in runs_list.iterrows():
        graph.insert_run(run_id=run['run_id'])

    graph.close()
    print("All runs have been inserted into the graph.")

    vectorIndexing = VectorIndexing(
        uri=URI,
        auth=(USER, PASSWORD),
        db_name='neo4j',
        openai_token=TOKEN 
    )

    vectorIndexing.create_embeddings()

    close_postgresql(connection)

if __name__ == "__main__":
    # set database information and API keys in environment variables
    insert_runs_into_graph()