import asyncio, os, sys, json, openai, litellm
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .client_config import get_client_model
from openai import AsyncAzureOpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ontorag_logger as logger

SYSTEM_PROMPT = """
You are a helpful data analyst specialized on machine learning data who can interact with the connected Neo4j database containing runs from OpenML by using the given tools.
You will receive queries and commands from the user, and you should respond with the results. 
Use the specific tools to interact with the database, such as getting the schema and reading data.
You can use the following tools:
{tools}

"""

EXAMPLE_PROMPT = """
Here is an examples of how to interact with the database. To ensure correct usage, always retrieve the schema first:
USER INPUT: 'What are the best runs and its hyperparametersettings?'
1. Retrieve the schema then run the query to retrieve the best predictive accuracy for the dataset and return ONLY the name of dataset:
QUERY: MATCH (d:Dataset {name: 'creditg'}) MATCH (r:Run)-[:hasInput]->(d) MATCH (r)-[:hasOutput]->(me:ModelEvaluation) RETURN r, d.name, me ORDER BY me.hasValue DESC LIMIT 3

2. Get the hyperparametersettings for each of the runs:
QUERY: MATCH (r:Run {name: 'retrieved run name'})-[:hasInput]->(hps:HyperParameterSetting) RETURN hps.name, hps.hasValue

3. Get the used software and implementation of the run:
QUERY: MATCH (r:Run {name: 'retrieved run name'})-[:executes]->(i:Implementation) MATCH (i)-[:hasPart]->(s:Software) RETURN i.name, s.name, s.hasVersion

Do the extractions step by step, and do not try to extract all information at once.
First extract the relevant runs for the dataset, then extract the hyperparameter settings for those runs.
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
        # Pass current environment variables to the server process
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=dict(os.environ)
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
              "content": SYSTEM_PROMPT.format(tools=available_tools) + EXAMPLE_PROMPT
          },
          {
              "role": "user",
              "content": query
          }
      ]

      # TODO use litellm
      # client = AsyncAzureOpenAI(
      #     api_key=os.getenv("AZURE_API_KEY"),
      #     api_version="2025-01-01-preview",
      #     azure_endpoint=os.getenv("AZURE_ENDPOINT")
      # )


      # deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
      client, deployment = get_client_model()
      
      response = await client.chat.completions.create(
          model=deployment,
          max_tokens=1000,
          messages=messages,
          tools=available_tools
      )

      # Process response and handle tool calls
      final_text = []
      choices = response.choices

      # assistant_message_content = []
      for choice in choices:
          message = choice.message
          if message.content:
              final_text.append(message.content)
              logger.info(f"Assistant response: {message.content}")
          if getattr(message, "tool_calls", None):
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

              if response.choices[0].message.tool_calls:
                  choices.append(response.choices[0])

              if response.choices and response.choices[0].message.content:
                  final_text.append(response.choices[0].message.content)
               
              logger.debug("Messages after tool calls:" + str(messages))

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

    async def process_query_for_run(self, query: str) -> tuple[str, dict]:
      """Run a query to retrieve runs from the server and handle tool calls"""
      logger.info(f"Processing query for runs: {query}")
      tokens = {
          "completion_tokens": 0,
          "prompt_tokens": 0,
          "total_tokens": 0,
          "embedding_tokens": 0
      }
      response = await self.session.list_tools()
      first_message = True

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
              "content": SYSTEM_PROMPT.format(tools=available_tools) + EXAMPLE_PROMPT
          },
          {
              "role": "user",
              "content": query
          }
      ]

      response = litellm.completion(
          api_base=os.getenv("OPENAI_ENDPOINT"),
          api_key=os.getenv("OPENAI_API_KEY"),
          model="azure_ai/" + os.getenv("LLM_MODEL", "azure-gpt-4o-mini"),
          max_tokens=1500,
          messages=messages,
          tools=available_tools
      )

      # client = openai.OpenAI(
      #     api_key=os.getenv("OPENAI_API_KEY"),
      #     base_url="http://litellm.warhol.informatik.rwth-aachen.de"
      # )

      # model = os.getenv("LLM_MODEL", "gpt-4o-mini")

      # response = client.chat.completions.create(
      #     model=model,
      #     max_tokens=1500,
      #     messages=messages,
      #     tools=available_tools
      # )

      tokens["completion_tokens"] = response.usage.completion_tokens
      tokens["prompt_tokens"] = response.usage.prompt_tokens
      tokens["total_tokens"] = response.usage.total_tokens

      logger.debug(f"Response tokens: {tokens}")

      # Process response and handle tool calls
      final_text = []
      choices = response.choices

      # assistant_message_content = []
      for choice in choices:
          message = choice.message
          if message.content:
              if first_message:
                  # try to query again with the query

                  response = litellm.completion(
                      api_base=os.getenv("OPENAI_ENDPOINT"),
                      api_key=os.getenv("OPENAI_API_KEY"),
                      model="azure_ai/" + os.getenv("LLM_MODEL", "azure-gpt-4o-mini"),
                      max_tokens=1500,
                      messages=messages,
                      tools=available_tools
                  )

                  tokens["completion_tokens"] += response.usage.completion_tokens
                  tokens["prompt_tokens"] += response.usage.prompt_tokens
                  tokens["total_tokens"] += response.usage.total_tokens

              if response.choices[0].message.tool_calls:
                  choices.append(response.choices[0])
              
              if response.choices and response.choices[0].message.content:
                  final_text.append(response.choices[0].message.content)

              final_text.append(message.content)
              # logger.info(f"Assistant response: {message.content}")

          if getattr(message, "tool_calls", None):
              first_message = False
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

                  messages.append({
                      "role": "tool",
                      "content": result.content[0].text,
                      "tool_call_id": tool_call.id,
                  })

              response = litellm.completion(
                  api_base=os.getenv("OPENAI_ENDPOINT"),
                  api_key=os.getenv("OPENAI_API_KEY"),
                  model="azure_ai/" + os.getenv("LLM_MODEL", "azure-gpt-4o-mini"),
                  max_tokens=1500,
                  messages=messages,
                  tools=available_tools
              )

              tokens["completion_tokens"] += response.usage.completion_tokens
              tokens["prompt_tokens"] += response.usage.prompt_tokens
              tokens["total_tokens"] += response.usage.total_tokens

              logger.debug(f"Response tokens: {tokens}")

              if response.choices[0].message.tool_calls:
                  choices.append(response.choices[0])
              
              if response.choices and response.choices[0].message.content:
                  final_text.append(response.choices[0].message.content)

      return "\n".join(final_text), tokens

    async def query_for_run(self, query: str):
        """Run a query to retrieve runs from the server"""
        try:
            response, tokens = await self.process_query_for_run(query)
            logger.info("\n" + response)
            logger.info(f"Tokens used: {tokens}")
            return response, tokens
        except Exception as e:
            logger.info(f"Error processing query for runs: {str(e)}")
            return f"Error processing query for runs: {str(e)}", []

    # async def chat_loop(self):
    #     """Run an interactive chat loop"""
    #     logger.info("MCP Client Started!")
    #     logger.info("Type your queries or 'quit' to exit.")
        
    #     while True:
    #         try:
    #             query = input("\nQuery: ").strip()
                
    #             if query.lower() == 'quit':
    #                 break
                    
    #             response = await self.process_query(query)
                    
    #         except Exception as e:
    #             logger.info(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
