from rdflib import Graph, Namespace, Literal, RDF, RDFS
from rdflib_neo4j import Neo4jStoreConfig, Neo4jStore, HANDLE_VOCAB_URI_STRATEGY
from neo4j import GraphDatabase
import re, os, json
from utils import ontorag_logger as logger

MLS = Namespace("http://openmlrun.org#")

class OntologyGraphStoreRun:
    def __init__(self):
        auth = {'uri': os.getenv("NEO4J_URI"),
                'database': "neo4j",
                'user': os.getenv("NEO4J_USER"),
                'pwd': os.getenv("NEO4J_PASSWORD")}
        config = Neo4jStoreConfig(auth_data=auth,
                                       handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.IGNORE,
                                       batching=True,)
        self.graph = Graph(store=Neo4jStore(config=config))
        self.graph.open(config)
        self.driver = GraphDatabase.driver(auth['uri'], auth=(auth['user'], auth['pwd']))

    def _safe_name(self, name):
        return re.sub(r'\W|^(?=\d)', '_', str(name)).strip('_')
    
    def max_run_id(self):
        with self.driver.session() as session:
            result = session.run("MATCH (r:Run) RETURN r.name ORDER BY r.name DESC LIMIT 1")
            max_run = result.single().get("r.name")
            if max_run:
                max_run_id = re.search(r'run(\d+)', max_run).group(1)

            logger.info(f"Max run ID from graph: {max_run_id if max_run_id else 0}")
            return max_run_id

    # Run structure example:
    # {
    #   "dataset": {  
    #     "dataset_name": "iris",
    #     "description": "Iris dataset",
    #     "qualities": {
    #       "NumberOfInstances": 150,
    #       "NumberOfFeatures": 4
    #     }
    #   },
    #   "flow": {
    #     "implementation": "TreeClassifier",
    #     "software": "scikit-learn",
    #     "parameters": {
    #       "max_depth": 3,
    #       "min_samples_split": 2
    #     }
    #   },
    def insert_new_run(self, run):
        run_id = int(self.max_run_id())+1

        logger.info(f"Inserting new run with ID: run{run_id}")
        run = json.loads(run)

        dataset = run.get("dataset")
        flow = json.loads(run.get("flow"))
        # dname = self._safe_name(dataset.get("dataset_name"))
        # task_name = self._safe_name(run.get("task_name"))
        flname = self._safe_name(flow.get("implementation"))
        fldependencies = self._safe_name(flow.get("software", "NoDependencies"))
        eval_measure = self._safe_name(json.loads(run.get("evaluation")).get("measure"))
        eval_value = run.get("evaluation_value", 0.0)
        # qualities = dataset.get("qualities", {})
        # instances = qualities.get("NumberOfInstances", 0)
        # features = qualities.get("NumberOfFeatures", 0)
        # dataset_description = dataset.get("description", "")
        # run_description = run.get("run_details", "No description available")
        # estimation_procedure = self._safe_name(run.get("estimation_procedure", "UnknownProcedure"))

        flname_lower = flname.lower()
        if "logistic" in flname_lower:
            algorithm_name = "LogisticRegression"
        elif "randomforest" in flname_lower:
            algorithm_name = "RandomForest"
        elif "tree" in flname_lower:
            algorithm_name = "DecisionTree"
        elif "knn" in flname_lower:
            algorithm_name = "KNN"
        elif "svm" in flname_lower:
            algorithm_name = "SVM"
        elif "adaboost" in flname_lower:
            algorithm_name = "AdaBoost"
        elif "gradientboosting" in flname_lower:
            algorithm_name = "GradientBoosting"
        elif "xgboost" in flname_lower:
            algorithm_name = "XGBoost"
        elif "lightgbm" in flname_lower:
            algorithm_name = "LightGBM"
        elif "catboost" in flname_lower:
            algorithm_name = "CatBoost"
        elif "rotationforest" in flname_lower:
            # https://ieeexplore.ieee.org/document/1677518
            algorithm_name = "RotationForest"
        else:
            algorithm_name = "UnknownAlgorithm"

        # URIs
        run_uri = MLS[f"run{run_id}"]
        task_uri = MLS[f"Task{task_name}"]
        implementation_uri = MLS[flname]
        software_uri = MLS[fldependencies]
        algorithm_uri = MLS[algorithm_name]
        dataset_uri = MLS[f"Dataset{dname}"]
        eval_measure_uri = MLS[eval_measure]
        # estimation_procedure_uri = MLS[estimation_procedure]
        evaluation_spec_uri = MLS[f"evaluationSpecification{run_id}"]
        model_eval_uri = MLS[f"modelEvaluation{run_id}"]
        model_uri = MLS[f"{flname}Model{run_id}"]
        # num_instances_uri = MLS[f"numberOfInstances{dname}"]
        # num_features_uri = MLS[f"numberOfFeatures{dname}"]
        print(run_uri)

        # Nodes
        self.graph.add((run_uri, RDF.type, MLS.Run))
        self.graph.add((run_uri, MLS.uri, Literal(str(run_uri))))
        self.graph.add((run_uri, RDFS.comment, Literal(run_description)))

        self.graph.add((task_uri, RDF.type, MLS.Task))
        self.graph.add((task_uri, MLS.uri, Literal(str(task_uri))))

        self.graph.add((implementation_uri, RDF.type, MLS.Implementation))
        self.graph.add((implementation_uri, MLS.uri, Literal(str(implementation_uri))))

        self.graph.add((software_uri, RDF.type, MLS.Software))
        self.graph.add((software_uri, MLS.uri, Literal(str(software_uri))))

        self.graph.add((algorithm_uri, RDF.type, MLS.Algorithm))
        self.graph.add((algorithm_uri, MLS.uri, Literal(str(algorithm_uri))))

        self.graph.add((dataset_uri, RDF.type, MLS.Dataset))
        self.graph.add((dataset_uri, MLS.uri, Literal(str(dataset_uri))))
        self.graph.add((dataset_uri, RDFS.comment, Literal(dataset_description)))

        self.graph.add((estimation_procedure_uri, RDF.type, MLS.EvaluationProcedure))
        self.graph.add((estimation_procedure_uri, MLS.uri, Literal(str(estimation_procedure_uri))))

        self.graph.add((eval_measure_uri, RDF.type, MLS.EvaluationMeasure))
        self.graph.add((eval_measure_uri, MLS.uri, Literal(str(eval_measure_uri))))

        self.graph.add((evaluation_spec_uri, RDF.type, MLS.EvaluationSpecification))
        self.graph.add((evaluation_spec_uri, MLS.uri, Literal(str(evaluation_spec_uri))))

        self.graph.add((model_eval_uri, RDF.type, MLS.ModelEvaluation))
        self.graph.add((model_eval_uri, MLS.uri, Literal(str(model_eval_uri))))
        self.graph.add((model_eval_uri, MLS.hasValue, Literal(eval_value)))

        self.graph.add((model_uri, RDF.type, MLS.Model))
        self.graph.add((model_uri, MLS.uri, Literal(str(model_uri))))

        self.graph.add((num_instances_uri, RDF.type, MLS.DatasetCharacteristic))
        self.graph.add((num_instances_uri, MLS.uri, Literal(str(num_instances_uri))))
        self.graph.add((num_instances_uri, MLS.hasValue, Literal(instances)))

        self.graph.add((num_features_uri, RDF.type, MLS.DatasetCharacteristic))
        self.graph.add((num_features_uri, MLS.uri, Literal(str(num_features_uri))))
        self.graph.add((num_features_uri, MLS.hasValue, Literal(features)))

        # Edges
        self.graph.add((run_uri, MLS.executes, implementation_uri))
        self.graph.add((run_uri, MLS.hasInput, dataset_uri))
        self.graph.add((run_uri, MLS.hasOutput, model_eval_uri))
        self.graph.add((run_uri, MLS.hasOutput, model_uri))
        self.graph.add((run_uri, MLS.realizes, algorithm_uri))
        self.graph.add((run_uri, MLS.achieves, task_uri))
        self.graph.add((implementation_uri, MLS.implements, algorithm_uri))
        self.graph.add((software_uri, MLS.hasPart, implementation_uri))
        self.graph.add((dataset_uri, MLS.hasQuality, num_features_uri))
        self.graph.add((dataset_uri, MLS.hasQuality, num_instances_uri))
        self.graph.add((model_eval_uri, MLS.specifiedBy, eval_measure_uri))
        self.graph.add((task_uri, MLS.definedOn, dataset_uri))
        self.graph.add((evaluation_spec_uri, MLS.defines, task_uri))
        self.graph.add((evaluation_spec_uri, MLS.hasPart, estimation_procedure_uri))
        self.graph.add((evaluation_spec_uri, MLS.hasPart, eval_measure_uri))

        # Hyperparameters
        for parameter, value in flow.get("parameters", {}).items():
            p = self._safe_name(parameter)
            hp_uri = MLS[f"{flname}{p}"]
            hps_uri = MLS[f"{flname}{p}Setting{run_id}"]
            self.graph.add((hp_uri, RDF.type, MLS.HyperParameter))
            self.graph.add((hp_uri, MLS.uri, Literal(str(hp_uri))))
            self.graph.add((hps_uri, RDF.type, MLS.HyperParameterSetting))
            self.graph.add((hps_uri, MLS.uri, Literal(str(hps_uri))))
            self.graph.add((hps_uri, MLS.hasValue, Literal(value if value is not None else "null")))
            self.graph.add((run_uri, MLS.has_input, hps_uri))
            self.graph.add((hps_uri, MLS.specifiedBy, hp_uri))
            self.graph.add((implementation_uri, MLS.hasHyperParameter, hp_uri))
        

        return run_id