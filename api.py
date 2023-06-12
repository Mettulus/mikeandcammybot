import logging
from typing import Any, Dict, List
import os
import asyncio
import time
import langchain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
import base64
from NPCFrameworkSDK.agents.npcagent import NPCAgent
from NPCFrameworkSDK.agents.baseagent import BaseAgent

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.callbacks import StdOutCallbackHandler
from NPCFrameworkSDK.customlang.intermediatesteps_callbackmanager import CustomStreamingStdOutCallbackHandler, CustomStreamlitCallbackHandler
from tuning_variables import *
from langchain.chat_models import ChatOpenAI
from NPCFrameworkSDK.modules.tools import Toolbelt
from NPCFrameworkSDK.modules.dataloader import IndexManager
import streamlit as st
#from streamlit_chat import message as st_message
import uuid


class StreamlitApp():

  def __init__(self):

    # Initialize the steamship environment
    #check_environment(RuntimeEnvironments.REPLIT)

    #self.init_agent()
    self.persona_files_path = data_folder + "persona_files"
    self.game_descriptions_path = data_folder + "game_descriptions"
    self.zaug_image = data_folder + "zaug3.png"
    self.zaug_background = "./zaug_background.png"
    self.manager = CallbackManager(
      [CustomStreamlitCallbackHandler(st.session_state)])

  def chaos_arcade_page(self, user_id):
    #st.set_page_config(layout="centered")
    # Read and encode the local image
    with open("zaug_background.png", "rb") as img_file:
      img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
    st.write(f"""
    <style>
        .stApp {{
            background-color: #1B1B1B;
            color: #E53935;
        }}
        .stApp {{
        background-color: #1B1B1B;
        color: #E53935;
        /* Add your image URL below */
        background-image: url('data:image/png;base64,{img_b64}');
        background-size: cover; /* Resize the background image to cover the entire container */
        background-position: center center; /* Center the image within the container */
    }}
    html[data-testid="stMarkdownContainer"] {{
        color: inherit;
    }}
    .centered {{
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    .caption {{
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: smaller;
    }}
    .message-container {{
    background-color: #3F3F3F; /* Choose your desired background color */
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    }}
</style>
""",
             unsafe_allow_html=True)

    l1, center, r1 = st.columns(3)
    with center:
      title = st.header('Chaos Arcade')
      st.markdown('<div class="centered"><h1>[🌀 AI]</h1></div>',
                  unsafe_allow_html=True)

      image = st.image(self.zaug_image, width=200)

      st.markdown('<div class="caption">Zaug: Slayer of Joy</div>',
                  unsafe_allow_html=True)

    st.divider()

    left_column, middle_column, right_column = st.columns(3)
    # You can use a column just like st.sidebar:

    persona_files = [
      f for f in os.listdir(self.persona_files_path)
      if os.path.isfile(os.path.join(self.persona_files_path, f))
    ]
    persona_names = [os.path.splitext(f)[0] for f in persona_files]

    persona_name_to_file = dict(zip(persona_names, persona_files))

    with left_column:
      st.session_state.ai_model_name = st.selectbox("AI Model",
                                                    ["gpt-4", "gpt-3.5-turbo"])
    with right_column:
      st.caption("Customize")
      with st.expander("Options"):
        st.caption("Values Are Updated On Each New Game Selection")
        temp = st.slider("Creativity", 0, 100, value=100)
        decimal_value = temp / 100.0
        st.session_state.temperature = decimal_value
        st.session_state.frequency_penalty = st.slider(
          "Frequency Penalty (1 = high repetition - 0 = low repetition)",
          0.0,
          1.0,
          value=0.0,
          step=0.1)

    # Or even better, call Streamlit functions inside a "with" block:
    with middle_column:
      chosen = st.radio('Games:', persona_names)
      st.write(f"Current Game: {chosen}")

    st.divider()
    # Grab game description and add to caption
    game_description_file = os.path.join(self.game_descriptions_path,
                                         f"{chosen}.txt")
    if os.path.exists(game_description_file):
      with open(game_description_file, 'r') as file:
        game_description = file.read()
        l1, m2, r2 = st.columns(3)

        with m2:
          st.subheader('Instructions')
        with st.container():
          st.markdown(
            f'<div class="message-container"><span style="color: #E53935;">Instructions</span><br/>{game_description}</div>',
            unsafe_allow_html=True)

    with st.form(key=f"{user_id}_username_form"):

      username_placeholder = st.empty()
      username = username_placeholder.text_input(
        "Please enter your username: ", key=f"{user_id}_username")

      username_submit_button = st.form_submit_button("Submit")

    st.divider()

    if username_submit_button:
      st.session_state.username = username

    if "username" not in st.session_state or st.session_state.username == "":
      st.warning("Please set a username before sending a message.")
    else:
      username_placeholder.empty()

      history = st.container()
      user_message_container = st.empty()
      user_message = user_message_container.text_area(
        " ", key=f"{user_id}_query_{len(st.session_state.history)}")
      submit_message_button = st.button('Send')

      thoughts = st.empty()
      self.manager.handlers[0].container = thoughts

      # Initialize the previous_chosen in session state if it doesn't exist
      if "previous_chosen" not in st.session_state:
        st.session_state.previous_chosen = ""
      # Find the index of the chosen persona name, and use it to load the corresponding file

      chosen_file = persona_name_to_file[chosen]

      if st.session_state.previous_chosen != chosen:

        self.init_agent(chosen_file)
        st.session_state.previous_chosen = chosen

      if st.session_state.ai_model_name != st.session_state.ai_model_name_last:
        self.init_agent(chosen_file)
        st.session_state.ai_model_name_last = st.session_state.ai_model_name

      with history:

        if user_message or submit_message_button:
          if "username" in st.session_state:

            spinner_container = st.container()

            with spinner_container:

              with st.spinner("Thinking..."):
                message_bot = self.query_agent(" " +
                                               st.session_state.username +
                                               ": " + user_message)

            st.session_state.history.append({
              "message": user_message,
              "is_user": True,
              "user_id": username
            })
            st.session_state.history.append({
              "message":
              message_bot,
              "is_user":
              False,
              "user_id":
              self.agent.master_ai_prefix
            })
            #st.session_state[f"{user_id}_query"] = ""

            #user_message_container.empty()
            #user_message = user_message_container.text_input(" ", key=f"{user_id}_query")
            user_message_container.empty()
            user_message_container.text_area(
              " ", key=f"{user_id}_query_{len(st.session_state.history)}")
        else:
          #st.write("Please set a username before sending a message.")
          pass

        for i, message in enumerate(st.session_state.history):
          with st.container():
            c1, c2 = st.columns([2, 10])
            with c1:
              st.write(message["user_id"])
            with c2:
              # Italicize the messages that are sent in the state
              if message["is_user"] == False:
                content = f'<span style="color: green;">{message["message"]}</span>'
              elif (len(st.session_state.history) - i <
                    st.session_state.context_length + 1):

                content = f'{message["message"]}'
              else:
                content = f'{message["message"]}'
              st.markdown(f'<div class="message-container">{content}</div>',
                          unsafe_allow_html=True)

  def custom_npc_page(self, user_id):

    # Add the content for the Custom NPC page here
    # You can use a similar structure as chaos_arcade_page but with the appropriate content for the Custom NPC page
    st.title("Custom NPC")

    character_name = st.text_input("Enter a character name:")

    files = st.file_uploader("Upload character files (scripts, notes, etc)",
                             type=["txt", "pdf"],
                             accept_multiple_files=True)
    generate_button = st.button("Generate")
    if (generate_button):
      self.init_custom_agent(files, character_name)

  # Add a select_page function to select between pages
  def select_page(self, user_id):

    sidebar = st.sidebar.title("Navigation")
    with sidebar:
      selected_page = st.radio(
        "Navigation:", ["Chaos Arcade", "Custom NPC (Under Construction)"])

    if selected_page == "Chaos Arcade":
      return self.chaos_arcade_page(user_id)
    elif selected_page == "Custom NPC (Under Construction)":
      return self.custom_npc_page(user_id)
    else:
      raise ValueError(f"Unsupported page: {selected_page}")

  def main(self, user_id):

    #self.chaos_arcade_page(user_id)
    self.select_page(user_id)

  def query_agent(self, query: str) -> str:

    message_bot, _ = self.agent.master_brain.query(
      query)  # Unpack the response and discard the intermiadiate steps
    return message_bot

  def query_custom_agent(self, query: str) -> str:

    message_bot, _ = self.custom_agent.master_brain.query(
      query)  # Unpack the response and discard the intermiadiate steps
    return message_bot

  def init_llm(self):

    self.llm = ChatOpenAI(model_name=st.session_state.ai_model_name,
                          temperature=st.session_state.temperature,
                          callback_manager=self.manager,
                          frequency_penalty=st.session_state.frequency_penalty,
                          streaming=True)

  def init_agent(self, persona_file_name_and_ext: str):
    self.init_llm()

    index_manager = IndexManager(data_folder, index_file_name,
                                 default_index_type, embedd_split_size)
    tools = Toolbelt(index_manager)
    toolkit = tools.chatnpc_toolkit

    agent = NPCAgent(master_data_dir=data_folder,
                     character_name=ai_name,
                     embedd_split_size=embedd_split_size,
                     keywords=keywords,
                     temperature=st.session_state.temperature,
                     frequency_penalty=st.session_state.frequency_penalty,
                     toolkit=toolkit,
                     tools=tools,
                     llm=self.llm)
    self.agent = agent
    print("Loading agent...")
    self.brain, error_message = self.load_agent_with_character_file(
      data_folder + "persona_files", persona_file_name_and_ext, self.agent)
    if error_message:
      print(error_message)

  def init_custom_agent(self, uploaded_documents, agent_name: str, keys=[""]):
    self.init_llm()

    index_manager = IndexManager(data_folder, index_file_name,
                                 default_index_type, embedd_split_size,
                                 uploaded_documents)
    tools = Toolbelt(index_manager)
    toolkit = tools.chatnpc_toolkit

    agent = NPCAgent(master_data_dir=data_folder,
                     character_name=agent_name,
                     embedd_split_size=embedd_split_size,
                     keywords=keys,
                     temperature=st.session_state.temperature,
                     frequency_penalty=st.session_state.frequency_penalty,
                     toolkit=toolkit,
                     tools=tools,
                     llm=self.llm)

    self.custom_agent = agent
    print("Loading agent...")
    self.custom_brain, error_message = asyncio.run(
      agent.load("", st.session_state.ai_model_name, uploaded_documents)), None
    if error_message:
      print(error_message)

  @staticmethod
  def load_agent_with_character_file(data_folder: str, filename: str,
                                     agent: BaseAgent):
    try:
      default_character_data = BaseAgent.load_character_file(
        data_folder, filename)
    except FileNotFoundError:
      return None, f"File '{filename}' not found in the data folder '{data_folder}'. Please check the file name and try again."
    return asyncio.run(agent.load(default_character_data, model_name)), None

  @staticmethod
  def load_agent(data_folder: str, filename: str, agent: BaseAgent):
    try:
      default_character_data = BaseAgent.load_character_file(
        data_folder, filename)
    except FileNotFoundError:
      return None, f"File '{filename}' not found in the data folder '{data_folder}'. Please check the file name and try again."
    return asyncio.run(
      agent.load(model_name=st.session_state.ai_model_name)), None

  @staticmethod
  def load_character_file(uploaded_files: List[str]) -> str:
    character_data = ""

    for uploaded_file in uploaded_files:
      if uploaded_file.type == "text/plain":
        character_data += uploaded_file.getvalue().decode("utf-8")
      elif uploaded_file.type == "application/pdf":
        # extract text from pdf and add it to character_data
        pass

    return character_data.strip()


def run_app():

  if 'app' not in st.session_state:
    st.session_state.app = StreamlitApp()
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.history = []
    st.session_state.context_length = 10

  if 'app' in st.session_state:
    if 'ai_model_name' not in st.session_state:
      st.session_state.ai_model_name = "gpt-4"
    if 'ai_model_name_last' not in st.session_state:
      st.session_state.ai_model_name_last = st.session_state.ai_model_name

    if 'temperature' not in st.session_state:
      st.session_state.temperature = 1

    if 'frequency_penalty' not in st.session_state:
      st.session_state.frequency_penalty = 0
    st.session_state.app.main(st.session_state['user_id'])


run_app()
