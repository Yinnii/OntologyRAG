import asyncio, os, sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import AsyncAzureOpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ontorag_logger as logger

SYSTEM_PROMPT = """
You are a helpful AI assistant that can interact with a Neo4j database using the MCP protocol.
You can execute queries, retrieve data, and perform various operations on the graph database.
You will receive queries and commands from the user, and you should respond with the results. 
Use specific tools to interact with the database, such as getting the schema and reading data.
Do not write any code or perform actions outside of the provided tools.
You can use the following tools:
{tools}
"""


SCHEMA_PROMPT = """
  Node properties:
  :`Run` {uri: ['String'], description: ['String'], name: ['String']}
  :`Task` {uri: ['String'], name: ['String']}
  :`Implementation` {uri: ['String'], name: ['String']}
  :`Software` {uri: ['String'], name: ['String']}
  :`Algorithm` {uri: ['String'], name: ['String']}
  :`Dataset` {uri: ['String'], description: ['String'], name: ['String']}
  :`EvaluationProcedure` {uri: ['String'], name: ['String']}
  :`EvaluationMeasure` {uri: ['String'], name: ['String']}
  :`EvaluationSpecification` {uri: ['String'], name: ['String']}
  :`ModelEvaluation` {uri: ['String'], hasValue: ['Double'], name: ['String']}
  :`Model` {uri: ['String'], name: ['String']}
  :`DatasetCharacteristic` {uri: ['String'], hasValue: ['Double'], name: ['String']}
  :`HyperParameter` {uri: ['String'], name: ['String']}
  :`HyperParameterSetting` {uri: ['String'], hasValue: ['String'], name: ['String']}

  Relationships:
  (:Run)-[:executes]->(:Implementation)
  (:Run)-[:hasInput]->(:Dataset)
  (:Run)-[:hasOutput]->(:ModelEvaluation)
  (:Run)-[:hasOutput]->(:Model)
  (:Run)-[:realizes]->(:Algorithm)
  (:Run)-[:achieves]->(:Task)
  (:Implementation)-[:implements]->(:Algorithm)
  (:Software)-[:hasPart]->(:Implementation)
  (:Dataset)-[:hasQuality]->(:DatasetCharacteristic)
  (:ModelEvaluation)-[:specifiedBy]->(:EvaluationMeasure)
  (:Task)-[:definedOn]->(:Dataset)
  (:EvaluationSpecification)-[:defines]->(:Task)
  (:EvaluationSpecification)-[:hasPart]->(:EvaluationProcedure)
  (:EvaluationSpecification)-[:hasPart]->(:EvaluationMeasure)
  (:Run)-[:has_input]->(:HyperParameterSetting)
  (:HyperParameterSetting)-[:specifiedBy]->(:HyperParameter)
  (:Implementation)-[:hasHyperParameter]->(:HyperParameter)
"""

EXAMPLE_PROMPT = """
Here are some examples of how to interact with the database:
USER INPUT: 'What is the predictive accuracy for the run25673?'
QUERY: MATCH (d:Dataset {name: 'anneal'}) MATCH (r:Run {name: 'run25673'})-[:hasInput]->(d) MATCH (r)-[:hasOutput]->(me:ModelEvaluation) RETURN me.name, me.hasValue ORDER BY me.hasValue

USER INPUT: 'Get the best runs for the dataset creditg'
QUERY: MATCH (d:Dataset {name: 'creditg'}) MATCH (r:Run)-[:hasInput]->(d) MATCH (r)-[:hasOutput]->(me:ModelEvaluation) RETURN r.name, me.hasValue ORDER BY me.hasValue DESC LIMIT 10

USER INPUT: 'Get the best hyperparametersettings for the dataset creditg'
QUERY: MATCH (d:Dataset {name: 'creditg'}) MATCH (r:Run)-[:hasInput]->(d) MATCH (r)-[:hasOutput]->(me:ModelEvaluation) MATCH (r)-[:has_input]->(hps:HyperParameterSetting) RETURN hps.name, hps.hasValue, me.hasValue ORDER BY me.hasValue DESC LIMIT 10
"""


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        logger.info("Available tools:" + ", ".join(tool.name for tool in tools))

    async def process_query(self, query: str) -> str:
      response = await self.session.list_tools()

      available_tools = [{
          "type": "function",
          "function": {
              "name": tool.name,
              "description": tool.description,
              "parameters": tool.inputSchema
          }
      } for tool in response.tools]

      messages = [
          {   "role": "system",
              "content": SYSTEM_PROMPT.format(tools=available_tools) + SCHEMA_PROMPT + EXAMPLE_PROMPT
          },
          {
              "role": "user",
              "content": query
          }
      ]

      client = AsyncAzureOpenAI(
          api_key=os.getenv("AZURE_API_KEY"),
          api_version="2025-01-01-preview",
          azure_endpoint=os.getenv("AZURE_ENDPOINT")
      )
      deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

      response = await client.chat.completions.create(
          model=deployment,
          max_tokens=1000,
          messages=messages,
          tools=available_tools
      )

      # Process response and handle tool calls
      final_text = []

      # assistant_message_content = []
      for choice in response.choices:
          message = choice.message
          if message.content:
              final_text.append(message.content)
              logger.info(f"Assistant response: {message.content}")
          if getattr(message, "tool_calls", None):
              import json
              # Add the assistant message with all tool_calls
              messages.append({
                  "role": "assistant",
                  "content": message.content or "",
                  "tool_calls": [tc.model_dump() for tc in message.tool_calls]
              })
              # For each tool call, execute and append a tool message
              for tool_call in message.tool_calls:
                  tool_name = tool_call.function.name
                  tool_args = json.loads(tool_call.function.arguments)
                  logger.info(f"Tool call: {tool_name} with args {tool_args}")
                  # Execute tool call
                  result = await self.session.call_tool(tool_name, tool_args)
                  final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                  logger.info(f"Tool result: {result.content}")

                  messages.append({
                      "role": "tool",
                      "tool_call_id": tool_call.id,
                      "name": tool_name,
                      "content": result.content
                  })
              # Get next response from Azure OpenAI
              response = await client.chat.completions.create(
                  model=deployment,
                  max_tokens=1000,
                  messages=messages,
                  tools=available_tools
              )
              if response.choices and response.choices[0].message.content:
                  final_text.append(response.choices[0].message.content)

      return "\n".join(final_text)

    async def query(self, query: str):
        """Run one query to retrieve data from the server"""
        logger.info(f"Processing query: {query}")
        try:
            response = await self.process_query(query)
            logger.info("\n" + response)
            return response
        except Exception as e:
            logger.info(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"
    
    async def process_query_for_run(self, query: str) -> str:
      response = await self.session.list_tools()

      available_tools = [{
          "type": "function",
          "function": {
              "name": tool.name,
              "description": tool.description,
              "parameters": tool.inputSchema
          }
      } for tool in response.tools]

      messages = [
          {   "role": "system",
              "content": SYSTEM_PROMPT.format(tools=available_tools) + SCHEMA_PROMPT + EXAMPLE_PROMPT
          },
          {
              "role": "user",
              "content": query
          }
      ]

      client = AsyncAzureOpenAI(
          api_key=os.getenv("AZURE_API_KEY"),
          api_version="2025-01-01-preview",
          azure_endpoint=os.getenv("AZURE_ENDPOINT")
      )
      deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

      response = await client.chat.completions.create(
          model=deployment,
          max_tokens=1000,
          messages=messages,
          tools=available_tools
      )

      # Process response and handle tool calls
      final_text = []

      # assistant_message_content = []
      for choice in response.choices:
          message = choice.message
          if message.content:
              final_text.append(message.content)
              logger.info(f"Assistant response: {message.content}")
          if getattr(message, "tool_calls", None):
              import json
              # Add the assistant message with all tool_calls
              messages.append({
                  "role": "assistant",
                  "content": message.content or "",
                  "tool_calls": [tc.model_dump() for tc in message.tool_calls]
              })
              # For each tool call, execute and append a tool message
              for tool_call in message.tool_calls:
                  tool_name = tool_call.function.name
                  tool_args = json.loads(tool_call.function.arguments)
                  logger.info(f"Tool call: {tool_name} with args {tool_args}")
                  # Execute tool call
                  result = await self.session.call_tool(tool_name, tool_args)
                  final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                  logger.info(f"Tool result: {result.content}")

                  messages.append({
                      "role": "tool",
                      "tool_call_id": tool_call.id,
                      "name": tool_name,
                      "content": result.content
                  })
              # Get next response from Azure OpenAI
              response = await client.chat.completions.create(
                  model=deployment,
                  max_tokens=1000,
                  messages=messages,
                  tools=available_tools
              )
              if response.choices and response.choices[0].message.content:
                  final_text.append(response.choices[0].message.content)

      return "\n".join(final_text)

    async def query_for_run(self, query: str):
        """Run a query to retrieve runs from the server"""
        logger.info(f"Processing query for runs: {query}")
        try:
            response = await self.process_query_for_run(query)
            logger.info("\n" + response)
            return response
        except Exception as e:
            logger.info(f"Error processing query for runs: {str(e)}")
            return f"Error processing query for runs: {str(e)}"

    async def chat_loop(self):
        """Run an interactive chat loop"""
        logger.info("MCP Client Started!")
        logger.info("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                    
            except Exception as e:
                logger.info(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

# async def main():
#     if len(sys.argv) < 2:
#         logger.error("Usage: python client.py <path_to_server_script>")
#         sys.exit(1)
        
#     client = MCPClient()
#     try:
#         await client.connect_to_server(sys.argv[1])
#         await client.chat_loop()
#     finally:
#         await client.cleanup()

# if __name__ == "__main__":
#     asyncio.run(main())