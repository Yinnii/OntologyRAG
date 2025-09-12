import asyncio, os, sys, json
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
You are a helpful data analyst specialized on machine learning data who can interact with the connected Neo4j database containing runs from OpenML by using the given tools.
You will receive a json containing the run details. 
Store the received run details in the database matching the schema.
Use the specific tools to interact with the database, such as getting the schema and writing data.
You can use the following tools:
{tools}
"""
# Better example necessary or go for static implementation
EXAMPLE_PROMPT = """
Here is an example of how to store a run. To ensure correct usage, always retrieve the schema first:
User Input: '{
    "run": {
      "name": "run25673",
      "dataset": {  
        "dataset_name": "iris",
      },
      "flow": {
        "implementation": "TreeClassifier",
        "software": "sklearn",
        "hyperparametersettings": {
          "max_depth": 3,
          "min_samples_split": 2
        }
      },
      "evaluation": {
        "measure": "predictiveaccuracy",
        "value": 0.95
      }
    }
}'
1. Retrieve the schema first to understand the structure of the data.
2. Ensure that the run ID is unique and that the run ID is appended to the implementation, software, hyperparametersettings, and hyperparameter to ensure uniqueness.
3. Store the run details in the database based on the schema.
"""

class MCPStoreClient:
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

    async def store_run(self, run_details: str) -> str:
          """Run a query to retrieve runs from the server and handle tool calls"""
          logger.info(f"Store the following run: {run_details}")
          tokens = []
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
                  "content": str(run_details)
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
              max_tokens=1500,
              messages=messages,
              tools=available_tools
          )

          completion_token = response.usage.completion_tokens
          prompt_tokens = response.usage.prompt_tokens
          total_tokens = response.usage.total_tokens
          tokens.append({
              "id": response.id,
              "completion_tokens": completion_token,
              "prompt_tokens": prompt_tokens,
              "total_tokens": total_tokens
          })

          logger.info(f"Response tokens: {tokens}")

          # Process response and handle tool calls
          final_text = []
          choices = response.choices

          # assistant_message_content = []
          for choice in choices:
              message = choice.message
              if message.content:
                  if first_message:
                      # try to query again with the query

                      response = await client.chat.completions.create(
                          model=deployment,
                          max_tokens=1500,
                          messages=messages,
                          tools=available_tools
                      )

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

                  # Get next response from Azure OpenAI
                  response = await client.chat.completions.create(
                      model=deployment,
                      max_tokens=1500,
                      messages=messages,
                      tools=available_tools
                  )

                  completion_token = response.usage.completion_tokens
                  prompt_tokens = response.usage.prompt_tokens
                  total_tokens = response.usage.total_tokens
                  overall_tokens = {
                      "id": response.id,
                      "completion_tokens": completion_token,
                      "prompt_tokens": prompt_tokens,
                      "total_tokens": total_tokens
                  }

                  tokens.append(overall_tokens)
                  logger.info(f"Response tokens: {tokens}")

                  if response.choices[0].message.tool_calls:
                      choices.append(response.choices[0])
                  
                  if response.choices and response.choices[0].message.content:
                      final_text.append(response.choices[0].message.content)

          return "\n".join(final_text), tokens

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()