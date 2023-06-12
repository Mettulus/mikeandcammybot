from langchain.prompts import PromptTemplate
from langchain.agents import Tool, tool
from typing import Dict, List, Any
from langchain.memory.chat_memory import BaseChatMemory
from llama_index.indices.base import BaseGPTIndex
from NPCFrameworkSDK.agents.baseagent import BaseAgent, AgentBrain
from llama_index import SimpleDirectoryReader, GPTTreeIndex, GPTListIndex, GPTVectorStoreIndex, SummaryPrompt, GPTKeywordTableIndex, LLMPredictor
from langchain.llms import OpenAI, OpenAIChat
from langchain.callbacks.manager import CallbackManager
from NPCFrameworkSDK.customlang.intermediatesteps_callbackmanager import CustomStreamingStdOutCallbackHandler, CustomStreamlitCallbackHandler
from termcolor import colored
from langchain.chat_models import ChatOpenAI
from NPCFrameworkSDK.modules.memory_module import AgentMemoryModule
from NPCFrameworkSDK.modules.promptmodule import PromptModule
from NPCFrameworkSDK.modules.tools import Toolbelt
from NPCFrameworkSDK.modules.dataloader import IndexManager


class NPCAgent(BaseAgent):
  """
  Narrative Personal Character Agent (NPC)
    
  Agents are defined by 'Tools,' 'Indexes,' 'Memory,' and 'Brains.'
  
  Agent are composed of any number of 'Brains' which are basic Agents with a single 'toolkit' and 'prompt'.
  
  NPC Agents have a single sub-brain, the 'extraction brain', which collects collects character sheet information from a Vector Store Index ('GPTSimpleVectorIndex') at a directory, constructs a summary of that information, and adds it to the prompt. 
  """

  def __init__(self,
               master_data_dir: str,
               character_name: str,
               index_type=GPTVectorStoreIndex,
               embedd_split_size=4000,
               llm=OpenAI(),
               temperature=1,
               frequency_penalty=0,
               keywords=None,
               toolkit: List[Tool] = None,
               tools: Toolbelt = None):

    #toolkit is loaded as master toolkit
    if keywords is None:
      keywords = []

    super().__init__(master_data_dir, character_name, index_type,
                     embedd_split_size, llm, temperature, frequency_penalty,
                     toolkit, tools)

    self.persona_data_dir = master_data_dir
    self.keywords = keywords
    self.npctoolkit = toolkit

  # Method to make an NPC agent with character sheet and name
  def make_npc_agent(self,
                     charactersheet: str,
                     character_name: str,
                     model_name="gpt-3.5-turbo",
                     llm=None):

    if llm is None:
      self.master_llm = ChatOpenAI(
        model_name=model_name,
        temperature=self.master_temperature,
        callback_manager=CallbackManager([CustomStreamlitCallbackHandler()]),
        frequency_penalty=self.master_frequency_penalty)  # model_name="gpt-4",

    self.npcprompt, self.npc_input_variables = self.promptmodule.chatnpc_prompt(
      self.master_toolkit)
    self.npc_input_variables["name"] = character_name
    self.npc_input_variables["chat_history"] = []
    self.npc_input_variables["metaprompt"] = charactersheet

    m = self.memory_module.conv_window_memory
    return AgentBrain(self.npctoolkit,
                      self.npcprompt,
                      m,
                      self.memory_module,
                      ai_prefix=self.master_ai_prefix,
                      llm=self.master_llm,
                      input_variables_dictionary=self.npc_input_variables)

  # Overwriting base load method with asynchronous method to load the NPC agent
  async def load(self,
                 character_sheet="",
                 model_name="gpt-3.5-turbo",
                 uploaded_documents=None):
    """Intializes values. Call Load before making calls to the master brain for the first time."""
    #Load Index and tools
    #self.load_master_index_and_tools()

    # Load the character sheet
    print("Enter Load")
    if (uploaded_documents):
      self.persona_extraction_brain = self.make_custom_persona_sheet_extract_agent(
        uploaded_documents, self.persona_data_dir, keywords=self.keywords)

    else:
      self.persona_extraction_brain = self.make_persona_sheet_extract_agent(
        self.persona_data_dir, keywords=self.keywords)

    if character_sheet == "":
      metaprompt, i = await self.persona_extraction_brain.async_query(
        "Execute Task")

      if (uploaded_documents):
        entitiesextractionbrain = self.make_custom_entity_extract_agent(
          uploaded_documents, self.persona_data_dir, self.keywords)
      else:
        entitiesextractionbrain = self.make_entity_extract_agent(
          self.persona_data_dir, self.keywords)

      entities, i = await entitiesextractionbrain.async_query("Execute Task")
      self.character_sheet = metaprompt + "\n\n" + entities
    else:
      self.character_sheet = character_sheet

    persona_sheet = colored(self.character_sheet, "blue")

    #Create master brain with charactersheet
    self.master_brain = self.make_npc_agent(persona_sheet,
                                            self.master_ai_prefix, model_name,
                                            self.master_llm)
    return self.master_brain

  # Method to create a Character Sheet Extract Brain
  def make_custom_persona_sheet_extract_agent(self,
                                              uploaded_documents,
                                              data_dir: str,
                                              keywords=[],
                                              ai_prefix="Extraction Agent"):
    """Method to create a Data Extract Brain"""
    memorymodule = AgentMemoryModule(ai_prefix=ai_prefix)
    memoryobject = memorymodule.conv_window_memory
    #memoryobject.ai_prefix = ai_prefix
    #index = IndexManager(data_dir, index_file_name, self.master_index_type,
    #                     self.embedd_split_size).load()

    keywords_string = ', '.join(keywords)

    tools = Toolbelt(
      IndexManager(data_dir, "index.json", self.master_index_type,
                   self.embedd_split_size, uploaded_documents))
    extractions_tools = tools.get_character_extraction_tools([''])

    extractionprompt, extraction_input_variables = self.promptmodule.persona_extraction_prompt(
      extractions_tools)

    extraction_input_variables["name"] = self.master_ai_prefix
    extraction_input_variables["chat_history"] = []

    extract_llm = OpenAI(temperature=0,
                         callback_manager=CallbackManager(
                           [CallbackManager(handlers=None)]),
                         frequency_penalty=0,
                         max_tokens=2500)  #model_name="gpt-3.5-turbo",
    return AgentBrain(extractions_tools,
                      extractionprompt,
                      memoryobject,
                      memorymodule,
                      self.master_ai_prefix,
                      llm=extract_llm,
                      input_variables_dictionary=extraction_input_variables)

  def make_custom_entity_extract_agent(self,
                                       uploaded_documents,
                                       data_dir: str,
                                       keywords=[],
                                       ai_prefix="Extraction Agent"):

    memorymodule = AgentMemoryModule(ai_prefix=ai_prefix)
    memoryobject = memorymodule.conv_window_memory
    #index = IndexManager(data_dir, index_file_name, self.master_index_type,
    #                     self.embedd_split_size).load()
    tools = Toolbelt(
      IndexManager(data_dir, "index.json", self.master_index_type,
                   self.embedd_split_size, uploaded_documents))
    extractions_tools = tools.get_entity_extraction_tools(keywords)
    #extractions_tools = tools.get_keyword_database_extraction_tools(
    #index, keywords)

    extractionprompt, extraction_input_variables = self.promptmodule.entity_extraction_prompt(
      extractions_tools)

    keywords_string = ', '.join(keywords)

    extraction_input_variables["name"] = keywords_string
    extraction_input_variables["chat_history"] = []

    extract_llm = OpenAI(temperature=0,
                         callback_manager=CallbackManager(
                           [CallbackManager(handlers=None)]),
                         frequency_penalty=0,
                         max_tokens=2500)  #model_name="gpt-3.5-turbo",
    return AgentBrain(extractions_tools,
                      extractionprompt,
                      memoryobject,
                      memorymodule,
                      self.master_ai_prefix,
                      llm=extract_llm,
                      input_variables_dictionary=extraction_input_variables)
