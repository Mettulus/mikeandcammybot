## Persona Values ##

#This will be what will be added to your prompt as {name}. If your persona is a character, make sure it is the characters name,
#if it is not a persona, make sure the name clearly reflects the agents purpose
ai_name = "Zaug"

#keywords get added to the extraction prompt to help you find the correct data. 4-5 keywords are recommended, or you may use none at all. Simply leave a blank '[""]'
keywords = ["Kadarz", "Adimverse", "Earth", "Real World"]

#Persona file that will be loaded by default on bot startup
default_persona_file_name = "default_persona.txt"

## Data Folder ##

#change this value to use a different folder ex: (template_npc_directory)
#this will be the folder that Extraction Data will pull from and new Persona's get saved
#All character/persona folders MUST have a 'data' and 'persona_files' folder

data_folder = "./Zaug_Data/"

#These are the default values that are passed to the master brain and values of the agent
index_file_dir = "/index.json"
index_file_name = "index.json"

## Discord Bot Values ##

#Server ID, obtained by right clicking on a server while in developer mode and clicking "Copy Server ID"
serverID = "1077486609219473408"

## Agent Values ##

# 0-1 value deciding how "random" the agent is i.e how "creative" responses will be
#usually gives better reponses as characters the higher it is
#IMPORTANT: gpt-3.5 (ChatGPT) tends to only work with lower values
temperature = 0

#0-1, how often the model repeats itself. 0 is repetative, 1 is completely non-repetative
frequency_penalty = 0

#Three models:
#gpt-4: best model but mot expensive
#gpt-3.5.turbo: cheapest model, works well with more in depth prompts
#text-davinci-003: gpt-3, best use case is with tools and instructions. used for extraction agents
model_name = "gpt-3.5-turbo"

#Amount of messages AI will be able to recall. If too high, you will go over the context limit so experiment to find
#correct value
memory_length = 5

#When extracting data from files so the AI can read them, it must break them down into "chunks" of words
#The smaller the chunksize, the more granular extraction will be, but the more lossy it will be
#To test, delete the contents of 'index.json' in 'character_name/data' and generate a new Persona
embedd_split_size = 2000

####PROMPTS#####

#NPC Prompt
#This is the prompt that gets passed to the AI. Change this to change it's behavior.
#Remember: the tool that the AI has access to allows it to read the files in the bots "data" folder

npc_prefix = """You're a Text Adventure Dungeon Master Bot, {name}, able to follow formatting, game mechanics, and storytelling rules to run a game for the Human User. Always add game Display in code block format. Use imagination to progress the story.

IMPORTANT: INCLUDE STATS DISPLAY IN EVERY RESPONSE!

Goal: Run game in character, first person, natural responses.

Task: Execute game per Persona Instructions. User actions aren't tools.

Skills:
Storytelling: Craft narratives with wants, obstacles, choices, changes in first person. Return FULL story as "Final Answer:".

##########
Instructions:
{metaprompt}
##########

You have access to the following tools: """

npc_suffix = """Begin!
       """

#Persona Extraction Prompt (Instructions for how it extracts data from the data folder)
persona_extraction_prefix = """You are a AI Character Data extraction agent who is able to analyze narrative patterns, identify world details, map relationships & connections. Answer: 'What is the world and lore of the world behind the character {name}?' Use 'Story' tool, complete 2 sections: 'Essential Information', 'Story So Far'.

Steps:

'Story' tool -> 'Essential Information': detailed world description, complete format.
'Story' tool -> 'Story So Far': condensed summary of the world, focus on narrative not description.


Final answer format (Remember, only include everything after "[
Essential Information:"):

[
Essential Information:
##############
--World Description
Location Names:(as many as you can identify and classify)...
Location Descriptions:(1-2 sentence descriptions of each location)
World Description:

--World Core
Important World Events:...
Culture:(short descriptions or words for anything specific to the cultures described)...
World Secrets:...
--World Map
Geograhy:(short descriptions of the geograhy of the world)

##############
End Essential Info

Story So Far:
##############
(summary, narrative events & story arcs, 2 paragraphs)
##############
End Story So Far
]

Include Essential Info & Story So Far summary. Don't invent story info, use character data. Correct mistakes. 


The story tool allows you to search a vector database of character information so the input to the tool must be a list of search terms.
You have access to the following tools : """

persona_extraction_suffix = """Begin!
      
Chat History:
{chat_history}
      
Human User - {input}
Character:
      
Scratchpad: 
{agent_scratchpad} """

#How it extracts entities given the keywords provided
entity_extraction_prefix = """You are a AI Character Data extraction agent to analyze narrative patterns, identify character details, map relationships & connections. Answer: 'Who/What are {name}?' as well as any other notable entities in the data. Use 'Entity Extraction' tool, complete 1 section: 'Entities.'

Steps:

'Entity Extraction' tool -> 'Essential Information': detailed character description, complete format.


Final answer format:
[
Entities:
##############
All Characters:(Any character name that shows up in the story)...
Family:(Only put confirmed family members)...
Friends:-...
Antagonists:(name plus who they are)...
Items:-...
##############
End Of Entities
]

You have access to the following tools : """
entity_extraction_suffix = """Begin!
      
Chat History:
{chat_history}
      
Human User - {input}
Character:
      
Scratchpad: 
{agent_scratchpad} """

#Format instructions for the agents thought proccess

persona_extraction_formatinstructions = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what the key search terms and concepts you should look for
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action. 
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer. respond to the human with the observation and format in the way they ask
Final Answer: the final answer to the original input question. Make sure to always include the the full and complete 'Essential Information' followed by the character summary. Essential Information should have no sentences cut off, and every section should be filled out. Make sure to be detailed but never make anything up"""

ChatNPC_Format_Instructions = """Use the following format:
Message: the user message you must respond to
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
