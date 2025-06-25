import neo4j, os


URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
AUTH = (os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'password'))
DB_NAME = 'neo4j'
driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()

openAI_token = os.getenv("AZURE_API_KEY", None)

search_prompt = """
{
    "name": "04_titanic",
    "description": "",
    "target_col": "Survived",
    "metadata": {
        "NumberOfClasses": 2,
        "NumberOfFeatures": 12,
        "NumberOfInstances": 891,
        "NumberOfInstancesWithMissingValues": 708,
        "NumberOfMissingValues": 866,
        "NumberOfNumericFeatures": 7,
        "NumberOfSymbolicFeatures": 5
    },
    "df_head": " PassengerId  Survived  Pclass                                                Name    Sex  Age  SibSp  Parch           Ticket    Fare Cabin Embarked\n           1         0       3                             Braund, Mr. Owen Harris   male 22.0      1      0        A/5 21171  7.2500   NaN        S\n           2         1       1 Cumings, Mrs. John Bradley (Florence Briggs Thayer) female 38.0      1      0         PC 17599 71.2833   C85        C\n           3         1       3                              Heikkinen, Miss. Laina female 26.0      0      0 STON/O2. 3101282  7.9250   NaN        S\n           4         1       1        Futrelle, Mrs. Jacques Heath (Lily May Peel) female 35.0      1      0           113803 53.1000  C123        S\n           5         0       3                            Allen, Mr. William Henry   male 35.0      0      0           373450  8.0500   NaN        S"
}"""

search_query = '''
WITH genai.vector.encode($searchPrompt, 'AzureOpenAI', { token: $token, resource: $resource, deployment: 'text-embedding-3-small'}) AS queryVector
CALL db.index.vector.queryNodes('embedding', 5, queryVector)
YIELD node, score
RETURN node.name as name, node.description, score
'''
records, summary, _ = driver.execute_query(
    search_query, searchPrompt=search_prompt, token=openAI_token, resource='gpt-dbis',
    database_=DB_NAME)
print(f'Datasets whos data relate to `{search_prompt}`:')
for record in records:
    print(record['name'], f'score: {record["score"]:.4f}')