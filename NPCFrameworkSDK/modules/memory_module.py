from langchain.llms import OpenAI, OpenAIChat
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, CombinedMemory, ConversationSummaryBufferMemory, ConversationEntityMemory, ConversationBufferWindowMemory, VectorStoreRetrieverMemory
from tuning_variables import memory_length
from langchain.vectorstores.base import VectorStore, VectorStoreRetriever
from NPCFrameworkSDK.modules.dataloader import IndexManager
from llama_index import GPTVectorStoreIndex
from llama_index.readers import Document
from typing import List, Any
import inspect
from langchain.memory.chat_memory import BaseChatMemory, ChatMessageHistory


class AgentMemoryModule:
  """This module is used to store the different types of agent memory. Memory is passed to the LLM Chain on Brain creation.
       This class is also called to extract the intermediant steps in a Langchain agents thought process.
  """

  def __init__(self,
               memory_llm=OpenAI(),
               chat_history_key="chat_history",
               input_key="input",
               ai_prefix="AI"):

    self.ai_prefix = ai_prefix
    self.chat_history_key = chat_history_key
    self.input_key = input_key
    self.memory_llm = memory_llm

    #Basic Memory type that displays the entire agent and user chat history
    self.conv_memory = ConversationBufferMemory(
      memory_key=self.chat_history_key, input_key=self.input_key)

    #Basic Memory type that displays the entire agent and user chat history. This one is used by default
    self.conv_window_memory = ConversationBufferWindowMemory(
      memory_key=self.chat_history_key,
      input_key=self.input_key,
      k=memory_length,
      ai_prefix=self.ai_prefix)

    #Extracts entities from conversation history
    self.conv_entity_memory = ConversationEntityMemory(
      llm=self.memory_llm,
      input_key=self.input_key,
      chat_history_key=self.chat_history_key)

    #Custom made chat memory chain that summarizes chat history and concatinates it over a certain legnth
    self.chat_summary_memory = ConversationSummaryMemory(
      ai_prefix=self.ai_prefix,
      llm=self.memory_llm,
      memory_key=self.chat_history_key,
      input_key=self.input_key)

    #Single window memory for debugging
    self.single_window_memory = ConversationBufferWindowMemory(
      memory_key=self.chat_history_key,
      input_key=self.input_key,
      k=1,
      ai_prefix=self.ai_prefix)

    #chat memory summarization with no concatination
    self.defaualt_chat_summary_memory = ConversationSummaryMemory(
      llm=self.memory_llm,
      memory_key=self.chat_history_key,
      input_key=self.input_key)

    #Summarizer memory, recursively summarizes conversation history
    self.summary_buffer_memory = ConversationSummaryBufferMemory(
      llm=self.memory_llm,
      memory_key=self.chat_history_key,
      input_key=self.input_key,
      max_token_limit=2000)

    #vector store retrieval memory for 'infinte' memory (not used)

    #manager = IndexManager("Zaug_Data/", "history", CustomVectorStore, 2000)
    #index = manager.create_index([Document(text="Memory:")])
    #retriever = VectorStoreRetriever(vectorstore=index)
    #self.vector_store_retriever_memory = VectorStoreRetrieverMemory(
    #  memory_key=self.chat_history_key,
    #  input_key=self.input_key,
    #  retriever=retriever)

  def construct_conv_memory(self):
    return ConversationBufferWindowMemory(memory_key=self.chat_history_key,
                                          input_key=self.input_key,
                                          k=memory_length,
                                          ai_prefix=self.ai_prefix)

  @staticmethod
  def create_new_memory_with_same_input_variables(memory_object):
    memory_type = type(memory_object)
    fields = memory_type.__fields__

    # Add these lines to print the fields and memory_object attributes
    #print("fields:", fields)
    #print("memory_object attributes:", dir(memory_object))

    kwargs = {}
    for field_name, field in fields.items():
      if hasattr(memory_object, field_name):
        kwargs[field_name] = getattr(memory_object, field_name)
      elif field.default != inspect.Parameter.empty:
        kwargs[field_name] = field.default

    if 'chat_memory' in kwargs:
      kwargs['chat_memory'] = ChatMessageHistory(messages=[])

    print("kwargs:", kwargs)  # Add this line to print the kwargs dictionary

    return memory_type(**kwargs)


class CustomVectorStore(GPTVectorStoreIndex, VectorStore):
  """Custom vector store that stores the agent memory in a custom vector store."""

  def add_texts(self, texts: List[str]) -> List[str]:
    documents = [Document(text=text) for text in texts]
    return self.add_documents(documents)

  def from_texts(self, texts: List[str]) -> "VectorStore":
    documents = [Document(text=text) for text in texts]
    return self.from_documents(documents)

  def similarity_search(self, query: str, **kwargs: Any) -> List[Document]:
    return self.search(query, **kwargs)
