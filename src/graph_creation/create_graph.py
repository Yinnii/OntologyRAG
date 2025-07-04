from neo4j import GraphDatabase
from openml import runs, flows, datasets, tasks
import re

class OntologyGraph:

    ontology_path = "https://raw.githubusercontent.com/ML-Schema/core/master/mlMerge.ttl"
    mls_prefix = "http://openmlrun.org#"

    def __init__(self, uri, user, password):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def load_ontology(self):
        with self.driver.session() as session:
            # Load the ontology into Neo4j
            session.run("""MATCH (n) DETACH DELETE n;""")
            # session.run("""CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE; """)
            session.run("""
              CALL n10s.graphconfig.init({
                  handleVocabUris: "IGNORE",
                  handleMultival: "ARRAY"
              });
            """)
            session.run("""
                CALL n10s.onto.import.fetch("/home/yin/Projects/OntologyRAG/ontologies/mlMerge.ttl", "Turtle")
            """)
            print("Ontology loaded successfully.")

    def _clean_data(self, name):
       return re.sub(r'\W|^(?=\d)', '', name)

    def _retrieve_implementation_name(self, flow_name):
       return re.search(r'classifier=([^,)\s]+)', flow_name)

    def _retrieve_software(self, flow_dependencies):
        try:
          fldependencies = {}
          if flow_dependencies is not None:
              # flow.dependencies form string to list
              if isinstance(flow_dependencies, str):
                  if "==" in flow_dependencies or ">=" in flow_dependencies:
                    flow_dependencies = flow_dependencies.split("\n")
                  else:
                    flow_dependencies = flow_dependencies.split(",")

              # create a dict of dependencies with the library and version 
              for d in flow_dependencies:
                  if "==" in d:
                    lib = d.split("==")[0]
                    version = d.split("==")[1] if "==" in d else "latest"
                    fldependencies[lib] = version
                  elif ">=" in d:
                    lib = d.split(">=")[0]
                    version = d.split(">=")[1] if ">=" in d else "latest"
                    fldependencies[lib] = version
                  else:
                    lib = d.split("_")[0]
                    version = d.split("_")[1] if "_" in d else "latest"
                    fldependencies[lib] = version
              print(fldependencies)
          else:
              fldependencies["NoDependencies"] = "null"

          return fldependencies
    
        except Exception as e:
          print(f"Error processing flow dependencies: {e}")
          fldependencies = {"NoDependencies": "null"}
          return fldependencies
    
    def _retrieve_algorithm_name(self, flow_name):
        flow_name_lower = flow_name.lower()
        algorithm_map = {
            "logistic": "LogisticRegression",
            "randomforest": "RandomForest",
            "decisiontree": "DecisionTree",
            "knn": "KNN",
            "svm": "SVM",
            "adaboost": "AdaBoost",
            "gradientboosting": "GradientBoosting",
            "xgboost": "XGBoost",
            "lightgbm": "LightGBM",
            "catboost": "CatBoost",
        }
        for key, value in algorithm_map.items():
            if key in flow_name_lower:
              return value

    def insert_run(self, run_id):
        run = runs.get_run(run_id)
        flow = flows.get_flow(run.flow_id)
        dataset = datasets.get_dataset(run.dataset_id, download_qualities=False, download_data=False, download_features_meta_data=False)
        task = tasks.get_task(run.task_id)
        if 'pipeline' in flow.name.lower():
          flname = self._retrieve_implementation_name(flow.name)
        else:
          flname = self._clean_data(flow.name)

        dname = self._clean_data(dataset.name)

        fldependencies = self._retrieve_software(flow.dependencies)
        
        algorithm_name = self._retrieve_algorithm_name(flname)

        eval_measure = self._clean_data(task.evaluation_measure).lower() if task.evaluation_measure else "predictiveaccuracy"
        eval_value = run.evaluations[task.evaluation_measure] if task.evaluation_measure else "null"
        
        with self.driver.session() as session:
            session.run(f"""
                        MERGE (run{run_id}:Run {{name: 'run{run_id}'}}) SET run{run_id}.uri = $run_uri SET run{run_id}.description = $run_description
                        MERGE (task{run.task_id}:Task {{name: 'task{run.task_id}'}}) SET task{run.task_id}.uri = $task_uri
                        MERGE ({flname + str(run_id)}:Implementation {{name: '{flname + str(run_id)}'}}) SET {flname + str(run_id)}.uri = $implementation_uri
                        MERGE ({algorithm_name}:Algorithm {{name: '{algorithm_name}'}}) SET {algorithm_name}.uri = $algorithm_uri
                        MERGE ({dname}:Dataset {{name: '{dname}'}}) SET {dname}.uri = $dataset_uri SET {dname}.description = $dataset_description
                        MERGE ({task.estimation_procedure["type"]}:EvaluationProcedure {{name: '{task.estimation_procedure["type"]}'}}) SET {task.estimation_procedure["type"]}.uri = $estimation_procedure_uri
                        MERGE ({eval_measure}:EvaluationMeasure {{name: '{eval_measure}'}}) SET {eval_measure}.uri = $eval_measure_uri
                        MERGE (evaluationSpecification{task.id}:EvaluationSpecification {{name: 'evaluationSpecification{task.id}'}}) SET evaluationSpecification{task.id}.uri = $evaluationSpecification_uri
                        MERGE (modelEvaluation{run_id}:ModelEvaluation {{name: 'modelEvaluation{run.id}'}}) SET modelEvaluation{run_id}.uri = $modelEvaluation_uri
                        MERGE ({flname}Model{run_id}:Model {{name: '{flname}Model{run_id}'}}) SET {flname}Model{run_id}.uri = $model_uri
                        MERGE (numberOfInstances{dname}:DatasetCharacteristic {{name: 'numberOfInstances{dname}'}}) SET numberOfInstances{dname}.uri = $numberOfInstances_uri
                        MERGE (numberOfFeatures{dname}:DatasetCharacteristic {{name: 'numberOfFeatures{dname}'}}) SET numberOfFeatures{dname}.uri = $numberOfFeatures_uri
                        MERGE (run{run_id})-[:executes]->({flname + str(run_id)})
                        MERGE (run{run_id})-[:hasInput]->({dname})
                        MERGE (run{run_id})-[:hasOutput]->(modelEvaluation{run.id})
                        MERGE (run{run_id})-[:hasOutput]->({flname}Model{run_id})
                        MERGE (run{run_id})-[:realizes]->({algorithm_name})
                        MERGE (run{run_id})-[:achieves]->(task{run.task_id})
                        MERGE ({flname + str(run_id)})-[:implements]->({algorithm_name})
                        MERGE ({dname})-[:hasQuality]->(numberOfFeatures{dname})
                        MERGE ({dname})-[:hasQuality]->(numberOfInstances{dname})
                        MERGE (modelEvaluation{run_id})-[:specifiedBy]->({eval_measure})
                        MERGE (task{run.task_id})-[:definedOn]->({dname})
                        MERGE (evaluationSpecification{task.id})-[:defines]->(task{run.task_id})
                        MERGE (evaluationSpecification{task.id})-[:hasPart]->({task.estimation_procedure["type"]})
                        MERGE (evaluationSpecification{task.id})-[:hasPart]->({eval_measure})
                        SET modelEvaluation{run_id}.hasValue = $eval_value
                        SET numberOfInstances{dname}.hasValue = $instances
                        SET numberOfFeatures{dname}.hasValue = $features""", 
                        eval_value=eval_value, 
                        instances=dataset.qualities['NumberOfInstances'], 
                        features=dataset.qualities['NumberOfFeatures'], run_uri=self.mls_prefix + f"run{run_id}",
                        task_uri=self.mls_prefix + f"Task{run.task_id}", implementation_uri=self.mls_prefix + f"{flname + str(run_id)}",
                        algorithm_uri=self.mls_prefix+f"{algorithm_name}", 
                        dataset_uri=self.mls_prefix + f"Dataset{dname}", eval_measure_uri=self.mls_prefix + f"{eval_measure}",
                        dataset_description=dataset.description, run_description = run.run_details if run.run_details is not None else "No description available",
                        estimation_procedure_uri=self.mls_prefix + f"{task.estimation_procedure['type']}", evaluationSpecification_uri=self.mls_prefix + f"evaluationSpecification{run_id}",
                        modelEvaluation_uri=self.mls_prefix + f"modelEvaluation{run.id}", model_uri=self.mls_prefix + f"{flname}Model{run_id}",
                        numberOfInstances_uri=self.mls_prefix + f"numberOfInstances{dname}", numberOfFeatures_uri=self.mls_prefix + f"numberOfFeatures{dname}") 
          
            # parse the fldependencies dict to create software nodes and its values
            for lib, version in fldependencies.items():
                lib_clean = self._clean_data(lib)
                version_clean = version.strip('-')
                node_name = f"{lib_clean}{run_id}"
                node_uri = self.mls_prefix + node_name
                # First, merge the Software node and set its properties
                session.run(
                    """
                    MERGE (sw:Software {name: $node_name})
                    SET sw.uri = $node_uri, sw.hasVersion = $version
                    """,
                    node_name=node_name, node_uri=node_uri, version=version_clean
                )
                # Then, create the relationship from Implementation to Software
                session.run(
                    """
                    MATCH (fl:Implementation {name: $flname})
                    MATCH (sw:Software {name: $node_name})
                    MERGE (fl)-[:hasPart]->(sw)
                    """,
                    flname=flname+str(run_id), node_name=node_name
                )      
                
            for parameter in flow.parameters:
                p = re.sub(r'\W|^(?=\d)', '_', parameter)  # Replace non-alphanumeric characters and leading digits with underscores
                p = p.strip('_')  # Remove leading or trailing underscores
                session.run(f"""
                      MATCH (run:Run {{name: 'run{run_id}'}})-[:realizes]->(alg:Algorithm {{name: $algorithm_name}})
                      MATCH (fl:Implementation {{name: $flname}})-[:implements]->(alg)
                      MERGE (hp:HyperParameter {{name: $hp_id}}) SET hp.uri = $hyperparameter_uri
                      MERGE (hps:HyperParameterSetting {{name: $hps_id}}) SET hps.uri = $hyperparameter_setting_uri
                      MERGE (run)-[:hasInput]->(hps)
                      MERGE (hps)-[:specifiedBy]->(hp)
                      SET hps.hasValue = $set_value
                      MERGE (fl)-[:hasHyperParameter]->(hp)
                    """, 
                    run_id=f"run{run_id}", algorithm_name=algorithm_name, flname=flname+str(run_id),
                    hp_id=f"{p}{run_id}", hps_id=f"{p}Setting{run_id}",
                    hyperparameter_uri=self.mls_prefix + f"{flname}{p}{run_id}", set_value=flow.parameters[parameter] if flow.parameters[parameter] is not None else "null", 
                    hyperparameter_setting_uri=self.mls_prefix + f"{flname}{p}Setting{run_id}")
            session.run(f"""
                  MATCH (n:HyperParameterSetting) where n.hasValue = 'null'
                  DETACH DELETE n
                  """)
            # Create a unique constraint on the Run name
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (r:Run) REQUIRE r.name IS UNIQUE;
            """)

    def run_exists(self, run_id):
      """ Check if a run already exists in the graph database.
      Args:
          run_id (int): The ID of the run to check.
      Returns:
          bool: True if the run exists, False otherwise.
      """
      with self.driver.session() as session:
          result = session.run("MATCH (r:Run {name: $run_name}) RETURN r", run_name=f"run{run_id}")
          return result.single() is not None
    
