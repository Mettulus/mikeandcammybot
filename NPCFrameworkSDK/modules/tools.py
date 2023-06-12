from langchain.prompts import PromptTemplate
from NPCFrameworkSDK.customlang.customzeroshotagent import CustomZeroShotAgent
from langchain.agents import Tool, tool
from typing import Dict, List, Any
from langchain.memory.chat_memory import BaseChatMemory
from llama_index.indices.base import BaseGPTIndex
from langchain.llms import OpenAI, OpenAIChat
from langchain.chains import ConversationalRetrievalChain
from NPCFrameworkSDK.modules.dataloader import IndexManager


class Toolbelt:
  """'Tools' stores all toolkits (arrays of Tools) for a given index. 
  
  A tool is an abstraction of an "action" an agent can take and can be called on by agents within chains"""

  def __init__(self, index_manager: IndexManager):

    self.index_manager = index_manager
    self.index = index_manager.load()

    self.query_engine = self.index.as_query_engine()
    self.default_toolkit = [
      Tool(
        name="Continue Story",
        func=lambda q:
        "Respond final answer with known information",  #answer_chain.run(q),
        description=
        "useful for when you want to answer the question with chat history or information stored in the prompt. If you decide more context is neccesary, use another tool."
      )
    ]

    self.chatnpc_toolkit = [
      Tool(
        name="Search Long Term Memory",
        func=lambda q: self.index.as_query_engine(similarity_top_k=2).query(q),
        description=
        "use this when you are asked about any information not contained in 'Essential Information' such as specific story events and characters. ALWAYS format your response with the following format: Final Answer: [your action input as a response here]'. The input to this tool should be a complete english sentence along with a list of keywords to search for."
      )
    ]

  ######################
  # Tool Functions

  def get_character_extraction_tools(self, keywords):
    # Join the keywords with spaces and add them to the 'q' variable
    query_string = ', '.join(keywords)

    return [
      Tool(
        name="Story",
        func=lambda q: self.index.as_query_engine(similarity_top_k=3).query(
          str(q + ' ' + query_string)),
        description=
        "use this to search the character data for any relevant story or narrative information. The input to this tool must be a list of search terms based on the sub categories of Essential Information"
      ),
      Tool(
        name="Direct Response",
        func=lambda q: "Respond final answer",  # answer_chain.run(q),
        description="useful for when you want to directly respond to the human"
      )
    ]

  def get_entity_extraction_tools(self, keywords):
    # Join the keywords with spaces and add them to the 'q' variable
    query_string = ', '.join(keywords)

    return [
      Tool(
        name="Entity Extraction",
        func=lambda q: self.index.as_query_engine(similarity_top_k=5).query(
          (q + ' ' + query_string)),
        description=
        "use this to extract information about all entities in the character data. The input to this tool must be a list of search terms from the categories of 'Entities'"
      ),
      Tool(
        name="Direct Response",
        func=lambda q: "Respond final answer",  # answer_chain.run(q),
        description="useful for when you want to directly respond to the human"
      )
    ]

  def get_keyword_database_extraction_tools(self, index: BaseGPTIndex,
                                            keywords: [str]):

    query_string = ', '.join(keywords)
    query_engine = index.as_query_engine()
    return [
      Tool(
        name="Entity Extraction",
        func=lambda q: query_engine.query(
          'Search for all information on the following keywords: ' +
          query_string),
        description=
        "use this to extract information about all entities in the character data."
      ),
      Tool(
        name="Direct Response",
        func=lambda q: "Respond final answer",  # answer_chain.run(q),
        description="useful for when you want to directly respond to the human"
      )
    ]

  ####################################
  #  The following are the tools as ACTIONS that get passed as instructions that are used to build the character sheet or other tools

  @tool
  def Default_DirectAnswer(query: str) -> str:
    """Empty Prompt"""

    return "Respond final answer with known information"

  def load(self, index_manager: IndexManager):
    self.index = index_manager.load()
