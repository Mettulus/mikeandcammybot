from NPCFrameworkSDK.modules.memory_module import AgentMemoryModule
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI, OpenAIChat
from langchain.agents import initialize_agent, ZeroShotAgent, AgentExecutor, load_tools
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import TextLoader
from langchain.vectorstores.base import VectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PagedPDFSplitter
from llama_index.readers import Document
from langchain.text_splitter import CharacterTextSplitter, NLTKTextSplitter, SpacyTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import OutputParserException

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field, validator

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, CombinedMemory, ConversationSummaryBufferMemory, ConversationEntityMemory, ConversationBufferWindowMemory
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  AIMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from langchain.agents import initialize_agent

from langchain.callbacks.tracers import LangChainTracer
from langchain.schema import AgentAction, AgentFinish, BaseMessage

from llama_index import SimpleDirectoryReader, GPTTreeIndex, GPTListIndex, GPTVectorStoreIndex, SummaryPrompt, GPTKeywordTableIndex, LLMPredictor
from langchain.chat_models import ChatOpenAI
import os, time
from termcolor import colored
from langchain.agents import Tool, tool
import sys
import json
from llama_index.embeddings.openai import OpenAIEmbedding
import discord
import asyncio
import re
from langchain.chains.base import Chain
from typing import Dict, List, Any
from NPCFrameworkSDK.modules.dataloader import IndexManager
from langchain.memory.chat_memory import BaseChatMemory
from llama_index.indices.base import BaseGPTIndex
from NPCFrameworkSDK.modules.promptmodule import PromptModule
from NPCFrameworkSDK.modules.tools import Toolbelt
from NPCFrameworkSDK.modules.dataloader import IndexManager
from tuning_variables import index_file_name, model_name
from NPCFrameworkSDK.customlang.intermediatesteps_callbackmanager import CustomStreamingStdOutCallbackHandler, CustomCallbackHandler
from langchain.callbacks.streamlit import StreamlitCallbackHandler

embed_model = OpenAIEmbedding()


########Agents#######
# BaseAgent class
class BaseAgent:
  """
  This is the base class for all Agents.
    
  To create an agent subclass, inheret this class and override the `Load` method.
  
  Agents are defined by 'Tools,' 'Indexes,' 'Memory,' and 'Brains.'
  
  Agent are composed of any number of 'Brains' which are basic Agents with a single 'toolkit' and 'prompt'.
  
  Agents have a single 'Master Brain' which is the brain that is used to execute the Agent which is used to coordinate and use other brains as 'Tools.'  

  We use the 'Master' prefix to denote the values that get passed to the Master Brain
  """

  # BaseAgent constructor
  def __init__(self,
               master_data_dir: str,
               ai_prefix="AI",
               index_type=GPTVectorStoreIndex,
               embedd_split_size=2000,
               llm=OpenAI(),
               temperature=0,
               frequency_penalty=0,
               toolkit: List[Tool] = None,
               tools: Toolbelt = None):

    #default values
    self.master_data_dir = master_data_dir
    self.promptmodule = PromptModule()
    self.memory_module = AgentMemoryModule(ai_prefix=ai_prefix)
    self.master_memory_object = self.memory_module.conv_window_memory
    self.defaultinputvalues_dict = {"input": "", "chat_history": []}

    #master values
    self.master_ai_prefix = ai_prefix
    self.embedd_split_size = embedd_split_size
    self.master_index_type = index_type
    self.master_temperature = temperature  #controls the 'temperature' of the agent which is essential randomness
    self.master_frequency_penalty = frequency_penalty  #how often words are allowed to be repeated
    self.master_llm = llm
    #load tools
    if (toolkit is not None):
      self.master_tools = tools
      self.master_toolkit = toolkit
    else:
      self.load_master_index_and_tools()

  # Property decorator for master_brain
  @property
  def master_brain(self):
    return self._master_brain

  # Setter for master_brain
  @master_brain.setter
  def master_brain(self, value):
    # Set the master_brain and update the master_executor
    self._master_brain = value
    self.master_executor = self.master_brain.agent_executor
    return self._master_brain

  def load_master_index_and_tools(self):
    """Loads in the Index, Index Manager and Toolkit for the Master Brain"""
    self.master_index_manager = IndexManager(self.master_data_dir,
                                             index_file_name,
                                             self.master_index_type,
                                             self.embedd_split_size)

    self.master_index = self.master_index_manager.load()
    self.master_tools = Toolbelt(self.master_index_manager)
    self.master_toolkit = self.master_tools.default_toolkit

# ConstructBrain method to create an AgentBrain instance

  def ConstructBrain(self,
                     prompt: PromptTemplate,
                     memory_object: BaseChatMemory,
                     memory_module: AgentMemoryModule,
                     toolkit=List[Tool],
                     ai_prefix="AI",
                     llm=OpenAI(),
                     input_dictionary={
                       "input": "",
                       "chat_history": []
                     },
                     ismaster=False):
    """Constructs a Brain. Input Bool 'ismaster' to true to set the Master Brain values to the brain you construct."""
    # Create a new AgentBrain instance and set master values if ismaster is True
    brain = AgentBrain(toolkit, prompt, memory_object, memory_module,
                       ai_prefix, llm, input_dictionary)

    if ismaster:
      if prompt:
        self.master_prompt = prompt
      if memory_object:
        self.master_memory_object = memory_object
      if toolkit:
        self.master_toolkit = toolkit
      if ai_prefix:
        self.master_ai_prefix = ai_prefix
      if memory_module:
        self.memory_module = memory_module

      self.master_brain = brain

    return brain

  # Load method to create a master agent with loaded master values
  def load(self,
           toolkit: [Tool],
           prompt: PromptTemplate,
           input_dictionary={
             "input": "",
             "chat_history": []
           }):
    """Create our master agent with loaded master values. Make sure `load_master_index_and_tools` is called first. """

    #self.load_master_index_and_tools()
    self.master_toolkit = toolkit

    self.master_brain = AgentBrain(
      self.master_toolkit,
      prompt,
      self.master_memory_object,
      self.memory_module,
      ai_prefix=self.master_ai_prefix,
      temperature=self.master_temperature,
      frequency_penalty=self.master_frequency_penalty,
      llm=self.master_llm,
      input_dictionary=input_dictionary)
    return self.master_brain

  # Method to create a Character Sheet Extract Brain
  def make_persona_sheet_extract_agent(self,
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
      IndexManager(data_dir, index_file_name, self.master_index_type,
                   self.embedd_split_size))
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

  def make_entity_extract_agent(self,
                                data_dir: str,
                                keywords=[],
                                ai_prefix="Extraction Agent"):

    memorymodule = AgentMemoryModule(ai_prefix=ai_prefix)
    memoryobject = memorymodule.conv_window_memory
    #index = IndexManager(data_dir, index_file_name, self.master_index_type,
    #                     self.embedd_split_size).load()
    tools = Toolbelt(
      IndexManager(data_dir, index_file_name, self.master_index_type,
                   self.embedd_split_size))
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

  def update_memory(self, new_memory: BaseChatMemory):
    """Update the agent's memory with a new instance of the master memory."""
    self.master_memory_object = new_memory
    self.master_brain.memory_object = new_memory
    self.master_brain.memory_module = self.memory_module
    self.master_brain.llm_chain.memory = new_memory

  def reload_memory(self):
    """Update the agent's master memory with a new instance of the same type of memory it's currently using. Resets Chat History"""
    current_memory = self.master_memory_object
    # Copy input variable dictionary from current memory to new memory
    new_memory = self.memory_module.create_new_memory_with_same_input_variables(
      current_memory)

    self.memory_module.conv_window_memory = new_memory
    self.master_brain.memory_object = new_memory
    self.master_brain.llm_chain.memory = new_memory

  async def async_query_master_brain(self, query: str):
    response, intermediate_steps = await self.master_brain.async_query(query)
    return response, intermediate_steps

  @staticmethod
  def load_character_file(data_folder: str, filename: str):
    file_path = os.path.join(data_folder, filename)
    with open(file_path, "r") as file:
      character_data = file.read()
    return character_data


########MODULES########


class AgentBrain:

  # AgentBrain constructor
  def __init__(self,
               toolkit: List[Tool],
               prompt: PromptTemplate,
               memory_object: BaseChatMemory,
               memory_module: AgentMemoryModule,
               ai_prefix="AI",
               llm=OpenAI(),
               input_variables_dictionary={
                 "input": "",
                 "chat_history": []
               },
               return_intermediate=True):
    print(ai_prefix)
    self.ai_prefix = ai_prefix

    # Initialize instance variables
    self.input_variable_dictionary = input_variables_dictionary
    self.toolkit = toolkit
    self.allowed_tools = [tool.name for tool in self.toolkit]
    self.embed_model = OpenAIEmbedding()

    #The prompt, memory type and memory module are the immutable components of a particulr brain
    self.prompt = prompt
    self.memory_module = memory_module
    self.memory_object = memory_object

    #create llm chain by passing in our memory
    self.llm = llm
    self.llm_chain = self.create_LLM_Chain()

    self.agent_executor = self.construct_agent_executor(
      self.toolkit, return_intermediate)

  # create_LLM_Chain method to create an LLMChain instance for the AgentBrain
  def create_LLM_Chain(self):
    """LLM Chain the works specifically with ChatOpenAI"""
    if str(self.llm.model_name) == "gpt-3.5-turbo" or str(
        self.llm.model_name) == "gpt-4":
      print("Flagged: " + self.llm.model_name)

      system_prompt = SystemMessagePromptTemplate(prompt=self.prompt)

      system_reminder = HumanMessagePromptTemplate.from_template(
        "\n############\nThe above was the previous conversation history and your thought process "
        f"(but I haven't seen any of your thought process! I only see what "
        "you return as Final Answer):\n\nHuman's Name: {input}\n\n\n\nScratchpad:\n\n{agent_scratchpad}"
      )

      chat_history = SystemMessagePromptTemplate.from_template(
        "\n\nChat History:\n\n{chat_history}\n\n")

      format_reminder = SystemMessagePromptTemplate.from_template(
        "\n\nWhen you have a response to say to the Human, you want to return a Final Answer, or you have determined you have enough to give a response, you MUST use the format: \n\n'Thought: Do I need to use a tool? No\n\nAction [action to take, does NOT need to be a tool]\n\nAction Input [input for the action]\n\nFinal Answer: [your action input as a response here]'\n############\n\nYou may only return ONE final answer per response. Never write more than one Final Answer for any individual response. \nImportant: Do NOT write anything after you write a Final Answer, and never write anything other than 'Final Answer:' to send a response, or the human cannot see it. Anything in chat history not prefaced by 'Final Answer' the human has not seen. (NEVER simply write your response without formatting it first or it will cause an error.)"
      )

      messages = [
        format_reminder, system_prompt, chat_history, system_reminder
      ]

      manager = CallbackManager([CustomStreamingStdOutCallbackHandler()])

      prompt = ChatPromptTemplate.from_messages(messages)
      #prompt.output_parser = parser
      llm_chain = LLMChain(llm=self.llm,
                           prompt=prompt,
                           verbose=True,
                           memory=self.memory_object,
                           callback_manager=manager)
    else:
      print("Flagged 1: " + self.llm.model_name)

      llm_chain = LLMChain(llm=self.llm,
                           prompt=self.prompt,
                           verbose=True,
                           memory=self.memory_object)

    return llm_chain

  # async_query method for asynchronously querying the LLM
  async def async_query(self,
                        query: str,
                        max_retries=3,
                        return_intermediate_steps=False):
    """Async Query to the LLM."""
    # Perform an async query to the LLM with retrying in case of errors
    loop = asyncio.get_event_loop()
    query = query.strip()
    executor = self.agent_executor

    input_variables = self.input_variable_dictionary
    intermediate_steps = ""
    input_variables["input"] = query

    retries = 0
    while retries < max_retries:
      try:
        input_variables = self.input_variable_dictionary.copy()

        intermediate_steps = ''
        message = await loop.run_in_executor(None, executor, input_variables)
        if return_intermediate_steps:

          i = message["intermediate_steps"]
          intermediate_steps = ''.join(str(step) for step in i)
          intermediate_steps = self.format_intermediate_steps(
            intermediate_steps)
        print(f"Attempt {retries}/{max_retries}")
        return message["output"].strip(), intermediate_steps
      except (ValueError, OutputParserException) as e:

        print(f"Error: {e}")
        retries += 1
        print(f"Retrying... Attempt {retries}/{max_retries}")

    error_message = "**(There was an issue processing your request. Please try again.)**"
    return error_message, ''

  # query method for synchronously querying the LLM
  def query(self, query: str, max_retries=3, return_intermediate_steps=False):
    """Synchronous Query to the LLM."""
    # Perform a query to the LLM with retrying in case of errors
    query = query.strip()
    executor = self.agent_executor

    input_variables = self.input_variable_dictionary
    intermediate_steps = ""
    input_variables["input"] = query

    retries = 0
    while retries < max_retries:
      try:
        input_variables = self.input_variable_dictionary.copy()

        intermediate_steps = ''
        message = executor(input_variables)
        #print(executor.agent.ru.on_llm_new_token())

        if return_intermediate_steps:

          i = message["intermediate_steps"]
          intermediate_steps = ''.join(str(step) for step in i)
          intermediate_steps = self.format_intermediate_steps(
            intermediate_steps)
        print(f"Attempt {retries}/{max_retries}")
        return message["output"].strip(), intermediate_steps
      except (ValueError, OutputParserException) as e:

        print(f"Error: {e}")
        retries += 1
        print(f"Retrying... Attempt {retries}/{max_retries}")

    error_message = "**(There was an issue processing your request. Please try again.)**"
    return error_message, ''

  # extract_intermident_steps method to extract intermediate steps from the AgentBrain's thought process string
  def format_intermediate_steps(self, intermediate_steps: str):
    """This method is used to extract the intermediate steps of a Langchain agents thought process.
       This method is called to extract the intermediate steps of a Langchain agents thought process.
       
       """

    # Extract intermediate steps from the thought process string
    x = intermediate_steps
    actions = re.findall("Action: ([^\.]*)", x.strip())
    observations = re.findall("Observation: ([^\.]*)", x.strip())
    observations2 = re.findall("Observation - ([^\.]*)", x.strip())
    thought = re.findall(f"{self.ai_prefix.strip()}: ([^\.]*)", x.strip())

    output = ""
    if len(thought) > 0:
      output += "Thought: " + "\n".join(thought) + "\n".strip()
    if len(actions) > 0:
      output += "\nActions: " + "\n".join(actions) + "\n".strip()
    if len(observations) > 0:
      output += "\nObservation: " + "\n".join(observations) + "\n".strip()
    if len(observations2) > 0:
      output += "\nObservation: " + "\n".join(observations2) + "\n".strip()

    output = output.split("Final Answer: ")[0].strip()
    return output

  def construct_agent_executor(self,
                               toolkit: List[Tool],
                               return_intermediate=True):
    # ConstructAgentExecutor method to create an AgentExecutor instance for the AgentBrain
    manager = CallbackManager([CustomStreamingStdOutCallbackHandler()])
    agent = ZeroShotAgent(llm_chain=self.llm_chain,
                          tools=toolkit,
                          allowed_tools=self.allowed_tools,
                          return_intermediate_steps=return_intermediate,
                          callback_manager=manager)  #max_iterations=2

    return AgentExecutor.from_agent_and_tools(
      agent=agent,
      tools=toolkit,
      llm_chain=self.llm_chain,
      verbose=True,
      return_intermediate_steps=return_intermediate,
      streaming=True,
      callback_manager=manager)

  #takes two LLM chains and outputs their responses together
  class ConcatenateChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain

    @property
    def input_keys(self) -> List[str]:
      # Union of the input keys of the two chains.
      all_input_vars = set(self.chain_1.input_keys).union(
        set(self.chain_2.input_keys))
      return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
      return ['concat_output']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
      output_1 = self.chain_1.run(inputs)
      output_2 = self.chain_2.run(inputs)
      return {'concat_output': output_1 + output_2}
