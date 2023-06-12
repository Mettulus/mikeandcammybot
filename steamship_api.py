"""A Steamship package for answering questions with sources using Embeddings and LangChain.

To run it:
1. Get a Steamship API Key (Visit: https://steamship.com/account/api). If you do not
   already have a Steamship account, you will need to create one.
2. Copy this key to a Replit Secret named STEAMSHIP_API_KEY.
3. Click the green `Run` button at the top of the window (or open a Shell and type `python3 api.py`).

To see additional Steamship + Langchain examples, please visit our GitHub repo:
https://github.com/steamship-core/steamship-langchain

More information is provided in README.md.

To learn more about advanced uses of Steamship, read our docs at: https://docs.steamship.com/packages/using.html.
"""
import logging
from typing import Any, Dict, List
import os
import asyncio

import langchain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from steamship import Block, check_environment, File, RuntimeEnvironments, Steamship, Tag
from steamship.data.plugin.index_plugin_instance import SearchResult
from steamship.invocable import post, PackageService
from steamship_langchain.cache import SteamshipCache
from steamship_langchain.llms import OpenAI
from termcolor import colored

from NPCFrameworkSDK.agents.npcagent import NPCAgent
from langchain.callbacks.base import CallbackManager
from NPCFrameworkSDK.customlang.intermediatesteps_callbackmanager import CustomStreamingStdOutCallbackHandler
from tuning_variables import *
from langchain.chat_models import ChatOpenAI
from NPCFrameworkSDK.modules.tools import Toolbelt
from NPCFrameworkSDK.modules.dataloader import IndexManager


class NPCPackage(PackageService):

  def __init__(self, client=None, **kwargs):
    super().__init__(client, **kwargs)

    # set up LLM cache
    self.llm = None
    if client is not None:
      self.init_llm(client)

    langchain.llm_cache = SteamshipCache(self.client)

  @post("/query")
  def query(self, query: str) -> str:

    return self.agent.master_brain.query(query)

  @staticmethod
  def load_agent_with_character_file(data_folder: str, filename: str,
                                     agent: NPCAgent):
    try:
      default_character_data = NPCPackage.load_character_file(
        data_folder, filename)
    except FileNotFoundError:
      return None, f"File '{filename}' not found in the data folder '{data_folder}'. Please check the file name and try again."
    return asyncio.run(agent.load(default_character_data, model_name)), None

  @classmethod
  def create(cls, client):
    instance = cls(client=client)
    return instance

  @staticmethod
  def load_character_file(data_folder: str, filename: str):
    file_path = os.path.join(data_folder, filename)
    with open(file_path, "r") as file:
      character_data = file.read()
    return character_data

  def init_llm(self, client):
    self.llm = SteamshipChatOpenAI(client=self.client,
                                   cache=True,
                                   model_name=model_name,
                                   temperature=temperature,
                                   callback_manager=CallbackManager(
                                     [CustomStreamingStdOutCallbackHandler()]),
                                   frequency_penalty=frequency_penalty)

  def run(self):

    print("Entering run()")
    logging.getLogger().setLevel(logging.ERROR)
    print(colored("NPC\n", attrs=['bold']))

    # This helper provides runtime API key prompting, etc.
    check_environment(RuntimeEnvironments.REPLIT)

    if self.llm is None:
      self.init_llm(self.client)

    index_manager = IndexManager(data_folder, index_file_name,
                                 default_index_type, embedd_split_size)
    tools = Toolbelt(index_manager)
    toolkit = tools.chatnpc_toolkit

    agent = NPCAgent(master_data_dir=data_folder,
                     character_name=ai_name,
                     embedd_split_size=embedd_split_size,
                     keywords=keywords,
                     temperature=temperature,
                     frequency_penalty=frequency_penalty,
                     toolkit=toolkit,
                     tools=tools,
                     llm=self.llm)

    self.agent = agent
    #await self.agent.load("1")
    print("Loading agent...")
    self.brain, error_message = NPCPackage.load_agent_with_character_file(
      data_folder + "persona_files", default_persona_file_name, self.agent)

    if error_message:
      print(error_message)
    else:

      while True:  # Replace this with 'for _ in range(number_of_iterations):' for a specific number of iterations
        query = input()
        print(colored("\nQuery: ", "blue"), f"{query}")

        print(
          colored(
            "Awaiting results. Please be patient. This may take a few moments.",
            "blue"))

        response = self.query(query=query)
        print(colored("Answer: ", "blue"), f"{response}")


class SteamshipChatOpenAI(ChatOpenAI, OpenAI):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  async def _agenerate(self, *args, **kwargs):
    return await super(ChatOpenAI, self)._agenerate(*args, **kwargs)

  def _generate(self, *args, **kwargs):
    return super()._generate(*args, **kwargs)


def main():
  # NOTE: we use a temporary workspace here as an example.
  # To persist chat history across sessions, etc., use a persistent workspace.
  with Steamship.temporary_workspace() as client:
    api = NPCPackage.create(client=client)
    api.run()


if __name__ == "__main__":
  main()
