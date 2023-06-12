from llama_index.readers import Document
from langchain.text_splitter import CharacterTextSplitter, NLTKTextSplitter, SpacyTextSplitter, RecursiveCharacterTextSplitter

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, CombinedMemory, ConversationSummaryBufferMemory, ConversationEntityMemory, ConversationBufferWindowMemory

from llama_index import SimpleDirectoryReader, GPTTreeIndex, GPTListIndex, GPTVectorStoreIndex, SummaryPrompt, GPTKeywordTableIndex, LLMPredictor
from llama_index import StorageContext, load_index_from_storage

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
from llama_index.indices.base import BaseGPTIndex
from io import BytesIO
from base64 import b64decode
import pdfplumber
import random
import datetime
from uuid import uuid4

embed_model = OpenAIEmbedding()


class DataLoader:
  """Data loader extracts documents from folders at `data.`"""

  def __init__(self, data_folder):
    self.data_folder = data_folder

  @staticmethod
  def extract_documents(folder_path: str, required_exts=['.txt', '.pdf']):
    # ...
    """This is a custom for-loop that is used to load .txt and pdf documents from the folder and subfolders. Needs more work"""

    #initialize lists
    docs = []

    #initialize PDF Doc Loaders
    doc_list = SimpleDirectoryReader(folder_path,
                                     recursive=True,
                                     required_exts=required_exts).load_data()

    for doc in doc_list:
      l = Document.to_langchain_format(doc)
      docs.append(l)

    return docs

  def process_uploaded_files(self, uploaded_files: List[str]):
    docs = []

    for uploaded_file in uploaded_files:
      file_type = uploaded_file.type
      file_stream = BytesIO(b64decode(uploaded_file.getvalue()))

      if file_type == "text/plain":
        text = file_stream.read().decode("utf-8")
      elif file_type == "application/pdf":
        text = ""
        with pdfplumber.open(file_stream) as pdf:
          for page in pdf.pages:
            text += page.extract_text()

      doc = Document(title=uploaded_file.filename, text=text.strip())
      docs.append(doc)

    return docs

  def load_documents(self,
                     required_exts=['.txt', '.pdf'],
                     uploaded_files=None):
    if uploaded_files:
      docs = self.process_uploaded_files(uploaded_files)
    else:
      docs = self.extract_documents(self.data_folder, required_exts)

    return docs


class IndexManager:

  def __init__(self,
               data_dir: str,
               index_file_name: str,
               index_type: BaseGPTIndex,
               split_size=500,
               uploaded_documents=None):

    self.index_file_path = data_dir + "/data" + "/" + index_file_name
    self.data_dir = data_dir + "data"
    self.persona_dir = data_dir

    self.data_loader = DataLoader(data_dir)
    self.index_type = index_type
    self.embed_model = OpenAIEmbedding()
    self.split_size = split_size
    self.uploaded_documents = uploaded_documents

  @property
  def data_loader(self):
    return self._data_loader

  @data_loader.setter
  def data_loader(self, value):
    self._data_loader = value
    return self._data_loader

  #Method to create an index
  def create_index(self, docs: [Document], uploaded_files=False):

    if os.path.isfile(self.index_file_path) and os.stat(
        self.index_file_path).st_size != 0 and not uploaded_files:

      print("INDEX FOUND")

      with open(self.index_file_path, 'r') as file:
        i = file.read()
      #index = self.index_type.load_from_disk(self.index_file_dir)

      storage_context = StorageContext.from_defaults(persist_dir=self.data_dir)
      index = load_index_from_storage(storage_context)

      print("LOADING INDEX! ")

    else:

      #create index file

      print("CREATING INDEX at: " + self.index_file_path)

      if (uploaded_files):
        #os.makedirs(os.path.dirname("/user_data"), exist_ok=True)
        with (open("./Zaug_Data/data/test", 'w')) as f:
          f.close()

      else:

        open(self.index_file_path, 'w').close()
        print("Rebuild File: " + self.index_file_path)
      for i, doc in enumerate(docs):
        print(f"Getting embedding for document {i}")
        emb = embed_model.get_text_embedding(doc.text)
        #emb = embed_model.get_text_embedding(lang_split_text.page_content)

        print(f"Finished embedding for document {i}")

        #inner doc get embedding (here we can add any metadata as well)
        doc.embedding = emb
        #index.insert(doc)

      index = self.index_type.from_documents(docs)

      if uploaded_files:
        #index.save_to_disk(self.data_dir)
        index.storage_context.persist(persist_dir="./user_data")
        pass
      else:
        index.storage_context.persist(persist_dir=self.data_dir)
      #storage_context = StorageContext.from_defaults(persist_dir=self.data_dir)
      #storage_context.persist(persist_dir=self.data_dir)

      #index.save_to_disk(self.index_file_dir)
      #print("CREATING INDEX!")
      #We end up with a vector index with each chunk having its own embedding, and embedding for all documents
    if index:
      print("Index Created!")
    return index

  # Method to load index
  def load_index(self, uploaded_documents=None):
    text_splitter = RecursiveCharacterTextSplitter(

      # Set a really small chunk size, just to show.
      chunk_size=self.split_size,
      chunk_overlap=100,
      length_function=len,
    )
    isuploaded = False

    if (uploaded_documents):
      docs = self.data_loader.load_documents(uploaded_documents)
      eventid = datetime.datetime.now().strftime('%Y%m%d%H%M%S') + str(uuid4())

      self.index_file_path = "./user_data" + "/" + eventid + "index.json"
      isuploaded = True
    else:
      docs = self.data_loader.load_documents()

    docs = text_splitter.split_documents(docs)
    split_docs = []
    int = 0
    print("Splitting Documents: ")

    for i, lang_split_text in enumerate(docs):

      doc = Document.from_langchain_format(lang_split_text)
      split_docs.append(doc)
      int += 1
      print("Document: " + str(int))

    return self.create_index(split_docs, isuploaded)

  def save_index(self, index):
    open(self.index_file_path, 'w').close()
    print("Rebuild File: " + self.index_file_path)
    index.save_to_disk(self.index_file_path)

  # Method to delete an index
  def delete_index(self):
    if os.path.exists(self.index_file_path):
      os.remove(self.index_file_path)
      print(f"Index file '{self.index_file_path}' has been deleted.")
    else:
      print(f"Index file '{self.index_file_path}' not found.")

  def load(self):
    self.index = self.load_index(self.uploaded_documents)
    return self.index
