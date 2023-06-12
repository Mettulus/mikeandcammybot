from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import os, time
from termcolor import colored
from gpt_index import SimpleDirectoryReader, GPTTreeIndex, GPTListIndex, GPTVectorStoreIndex, SummaryPrompt, GPTKeywordTableIndex, LLMPredictor
import logging
import sys
from gpt_index.composability import ComposableGraph
import json
from gpt_index.embeddings.openai import OpenAIEmbedding
import discord
from NPCFrameworkSDK.agents.npcagent import NPCAgent
import asyncio
from tuning_variables import *

from discordbot import NPCBot, botmain

foxtrotdata = "./foxtrotdata"
agent = NPCAgent(foxtrotdata, "Foxtrot", embedd_split_size=2000)

test_keywords = ["Ian", "Katie", "Ordun", "Chaos"]
agent = NPCAgent("./Zaug_Data/",
                 "Zaug",
                 embedd_split_size=2000,
                 keywords=test_keywords)