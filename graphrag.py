from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.embeddings import OpenAIEmbeddings
import os
from neo4j_graphrag.llm import AzureOpenAILLM

llm = AzureOpenAILLM(
    model_name="gpt-4o-mini",
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),  
    api_version=os.getenv("API_VERSION"),  
    api_key=os.getenv("API_KEY"), 
)
# llm.invoke("say something")

# 1. Neo4j driver
URI = os.getenv("NEO4J_URI" ,"neo4j://localhost:7687")
AUTH = (os.getenv("NEO4J_USER","neo4j"), os.getenv("NEO4J_PASSWORD","password"))

INDEX_NAME = "index-name"

# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)

# 2. Retriever
# Create Embedder object, needed to convert the user question (text) to a vector
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize the retriever
retriever = VectorRetriever(driver, INDEX_NAME, embedder)

# 3. LLM
# Note: the OPENAI_API_KEY must be in the env vars
# llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})

# Initialize the RAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# Query the graph
query_text = "How do I do similarity search in Neo4j?"
response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
print(response.answer)