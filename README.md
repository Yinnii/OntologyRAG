# Ontology-RAG for AutoML

Ontology-RAG combines GraphRAG and a Machine Learning ontology. The used machine learning ontology is created by mapping the [ML Schema](https://github.com/ML-Schema/core/blob/master/MLSchema.ttl) from W3C with [MLOnto](https://osf.io/chu5q/files/). Currently it only retrieves the classification tasks from OpenML to annotate the ontology stored in Neo4j and is used to guide AutoML processes based on the dataset description and metadata information.
