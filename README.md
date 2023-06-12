# NPC-Framework (Narrative Personal Character Framework)

Hey Adim Arcade Guild Members! 🎉 Get ready to dive into the world of NPC-Framework, a powerful tool that allows you to create, load, save, and upload character personas for chatbots in a Discord server. Built on top of Langchain, this framework makes it easy for anyone to create unique and engaging chatbot experiences, no coding experience required!

Here's a fun and simple breakdown of how to use the NPC-Framework with your Discord server:

1. Set up your Discord bot and connect it to your server.
2. Use slash commands like `/loadcharacterpersona`, `/savecharacterpersona`, `/uploadcharacterpersona`, `/say`, and `/restartbot` to interact with the bot.
3. Customize your chatbot persona using the tuning_variables file.
4. Extend the functionality of the NPC Discord Bot by collaborating with others in the Adim Arcade discord or through Replit's bounty system.

Let's take a look at the Table of Contents for an overview:

1. Introduction
2. How to use the NPC Discord Bot in Discord
   - Load a Character Persona
   - Save a Character Persona
   - Upload a Character Persona
   - Send a Message to the Bot
   - Restart the Bot Instance
3. Extending the NPC Discord Bot
4. Setting Up Replit
   - Access the Project
   - Set Up Replit Secrets
   - Obtain OpenAI API Key
   - Obtain Discord Bot Secret
5. Setting up an NPC Bot for Discord
   - Prerequisites
   - Create a new Discord Application
   - Configure the Bot
   - Set Up OAuth2 Permissions
   - Grab the Client Secret and Application ID
   - Set Up Replit
   - Invite the Bot to Your Server
   - Run the Bot
6. NPC-Framework Guide
   - Overview
   - Agent Class, Agent Brain, and NPCAgent
     - BaseAgent
     - AgentBrain
     - NPCAgent
   - How to Use
   - Ideas for Extending

We hope you have a blast creating unique and captivating NPC characters for your Discord server! Let the creation commence! 🚀

Access the codebase here: https://replit.com/join/fyggxnxswv-mettulus

# NPC Discord Bot

## Introduction

Welcome to the NPC Discord Bot! This bot is built on top of Langchain and allows you to create, load, save, and upload character personas for chatbots in a Discord server. These personas can be used to guide conversations like a game, customize character information, or tweak generated personas. The NPC Discord Bot is designed to be easy to use and customize through a tuning_variables file. No coding experience is required to use this bot, but if you want to extend its functionality, you can do so through Replit's bounty system or by asking for help in the Adim Arcade discord.

## How to use the NPC Discord Bot in Discord

Before you can use the bot in Discord, make sure it is properly set up and connected to your server. Once it's ready, you can use the following slash commands to interact with the bot:

- `/loadcharacterpersona`: Load a character persona from the data directory.
- `/savecharacterpersona`: Save the current character persona to a local file.
- `/uploadcharacterpersona`: Upload a character persona text file.
- `/say`: Send a message to the bot and wait for its response.
- `/restartbot`: Restart the bot instance.

Please note that some commands require administrator permissions.

Here's a quick overview of how to use each command:

### Load a Character Persona

To load a character persona, use the `/loadcharacterpersona` command followed by the name of the persona you want to load. If you want to generate a new character persona, use "Generate New Character Persona" as the name.

### Save a Character Persona

To save a character persona, use the `/savecharacterpersona` command followed by the desired filename. The bot will save the current character persona to a local file with the specified name.

### Upload a Character Persona

To upload a character persona, use the `/uploadcharacterpersona` command and attach a text file containing the character persona data. The bot will save the uploaded persona and update its cache of available character personas.

### Send a Message to the Bot

To send a message to the bot, use the `/say` command followed by your message. The bot will process your message and respond accordingly. Keep in mind that there is a limit of 2 responses in the queue at any given time.

### Restart the Bot Instance

To restart the bot instance, use the `/restartbot` command. This command can be helpful if you encounter issues with the bot or want to reset its state.

## Extending the NPC Discord Bot

If you have an idea for extending the functionality of the NPC Discord Bot, you can use Replit's bounty system to pay other programmers a nominal amount to implement your desired features. This is a great way for those without coding skills to bring their ideas to life. Additionally, you can request help in the Adim Arcade discord to draft bounties for extension, which can be a great way to collaborate with others and make your NPC Discord Bot even better!


In the following sections, you will learn more about Replit and how to set up your Discord bot to further customize and enhance its capabilities. Explore the possibilities and unleash your creativity to create a unique and engaging NPC Discord bot experience!

# ##########################################

# Setting Up Replit

This guide will walk you through the setup process for the project, including setting up the required keys and secrets. The project is hosted on Replit.com.

## Step 1: Access the Project

1. Go to the project using the provided link: https://replit.com/join/fyggxnxswv-mettulus

After accessing the project using the provided link (to be added), you'll need to create your own copy of the project by forking it. Follow these simple steps to fork the Replit project:

1. Make sure you have a Replit account. If you don't have one yet, sign up for a Replit account and follow the on-screen instructions.

2. Once you're logged into your Replit account, click on the provided project link (to be added) to access the NPC-Framework project.

3. On the top-left corner of the project page, you'll find a "Fork" button. Click on it to create a copy of the project under your account.

4. After the forking process is complete, you'll be redirected to your newly created copy of the project.

You can now proceed with setting up the Replit secrets.

## Step 2: Set Up Replit Secrets

In order to protect sensitive information such as API keys, Replit offers a "Secrets" tab. We will use this feature to store our API keys securely.

1. In the Replit project, click on the "Secrets" tab (the lock icon) on the left side of the screen.
2. Add the following three secrets (keys) one by one:

   - OPENAI_API_KEY
   - DISCORD_BOT_SECRET

Each key should have a corresponding value which you will need to obtain as described in the next steps.

## Obtain OpenAI API Key

To get your OpenAI API key, follow one of these methods:

- If you have access to the OpenAI platform, go to https://platform.openai.com/account/, sign in, and click "View API Key". Copy the key and use it as the value for the `OPENAI_API_KEY` secret in Replit.
- If you are a member of the Adim Arcade Guild, you can obtain the key from your guildmaster. Once you have the key, use it as the value for the `OPENAI_API_KEY` secret in Replit.

## Obtain Discord Bot Secret

To get your Discord Bot Secret, refer to the previous tutorial on setting up an NPC bot for Discord. Once you have the Client Secret, use it as the value for the `DISCORD_BOT_SECRET` secret in Replit.

# ##########################################

# Setting up an NPC Bot for Discord

This README will guide you through the process of setting up an NPC bot on Discord, using Replit to host the bot. The bot will have the specified permissions and intents.

## Prerequisites

Before starting, make sure you have the following:

- A Discord account
- [Discord Developer Portal](https://discord.com/developers/applications) access
- A [Replit](https://replit.com/) account
- Basic knowledge of Python programming (optional, but helpful)

## Step 1: Create a new Discord Application

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and sign in with your Discord account.
2. Click the "New Application" button in the top right corner and provide a name for your application.
3. Click "Create" to proceed.

## Step 2: Configure the Bot

1. On the left sidebar, click the "Bot" tab.
2. Click the "Add Bot" button, then confirm by clicking "Yes, do it!".
3. Under the "Privileged Gateway Intents" section, enable both "SERVER MEMBERS INTENT" and "MESSAGE CONTENT INTENT".
4. Scroll down and click "Save Changes".

## Step 3: Set Up OAuth2 Permissions

1. On the left sidebar, click the "OAuth2" tab.
2. In the "Scopes" section, select the following:
   - bot
   - applications.commands
3. In the "Bot Permissions" section, select the following:
   - Webhook: Incoming
   - Voice: Speak, Connect
   - Text: Send Messages, Send TTS Messages, Read Message History, Use Slash Commands, Read Messages/View Channel, Manage Events, Manage Messages, Send Messages in Threads, Manage Threads

The generated URL in the "Scopes" section can be used to invite the bot to your server. Make sure to copy the URL as you will need it in Step 6.

## Step 4: Grab the Client Secret and Application ID

1. On the left sidebar, click the "General Information" tab.
2. Copy the "Client Secret" value (or reset it if needed).
3. Copy the "Application ID" value.

## Step 5: Set Up Replit

1. Go to [Replit](https://replit.com/) and sign in with your account.
2. Open NPC Project you forked in the previous guide.
3. In the `Secrets` tab, add the following key value pair:
   - DISCORD_BOT_SECRET
   - Client Secret from Step 4

To get the Guild ID for your Discord server, follow these steps:

1. Open Discord and go to your server.
2. Make sure you have "Developer Mode" enabled on your Discord account. To enable Developer Mode, go to "User Settings" (the gear icon in the lower-left corner) > "Appearance" > scroll down to the "Advanced" section, and toggle on "Developer Mode".
3. Right-click your server name at the top-left corner and click "Copy ID". The Guild ID is now copied to your clipboard.

For Reference, the Adim Arcade GuildId is: `1091086906189692999`

4. You can now use the copied Guild ID in the Python script.

Remember to replace `'your_guild_id'` with the actual Guild ID you just copied in `tuning_variables.py`:

```python

# Replace 'your_guild_id' with the actual Guild ID
serverID = "your_guild_id"

```

Make sure to import the `NPCBot` and `botmain` functions from your `discordbot` module and replace `'your_guild_id'` with the actual Guild ID.

## Step 6: Invite the Bot to Your Server

Use the bot invite link you copied in Step 3 to invite the bot to your server. Click the link, select the server you want to add the bot to, and click "Authorize".

## Step 7: Run the Bot

Run your Replit Python script to start and enter `/say`

# ##########################################

# NPC-Framework Guide

## Overview

This framework provides a foundation for building chatbots using Langchain. You can create chatbot personas by defining their "personas," which are sets of instructions for a conversation, custom character information, or tweaks to generated personas. It is designed for use with Discord bots and allows you to load, upload, generate, and save character personas. The main class for users to interact with is `NPCAgent`.

## Agent Class, Agent Brain, and NPCAgent

The framework consists of three main classes, `BaseAgent`, `AgentBrain`, and `NPCAgent`.

### BaseAgent

The `BaseAgent` class is the base class for all agents. To create an agent subclass, inherit from `BaseAgent` and override the `Load` method. Agents are composed of any number of 'Brains' which are basic Agents with a single 'toolkit' and 'prompt'. Agents have a single 'Master Brain' which is the brain that is used to execute the Agent, which is used alone or to coordinate and use other brains as 'Tools'. We use the 'Master' prefix to denote the values that get passed to the Master Brain.

#### Key Functions

- `load_master_index_and_tools`: Loads the Index, Index Manager, and Toolkit for the Master Brain.
- `ConstructBrain`: Constructs a Brain. Set the `ismaster` flag to true to set the Master Brain values to the brain you construct.
- `load`: Creates the master agent with loaded master values. Ensure `load_master_index_and_tools` is called first.
- `make_persona_sheet_extract_agent`: Creates a Data Extract Brain for persona sheets.
- `make_entity_extract_agent`: Creates a Data Extract Brain for entities.
- `async_query_master_brain`: Asynchronously queries the Master Brain.

### AgentBrain

The `AgentBrain` class is responsible for creating an LLMChain instance and provides a method for asynchronously querying the LLM.

#### Key Functions

- `create_LLM_Chain`: Creates an LLMChain instance for the AgentBrain.
- `async_query`: Asynchronously queries the LLM.
- `format_intermediate_steps`: Extracts intermediate steps from the AgentBrain's thought process string.
- `construct_agent_executor`: Creates an AgentExecutor instance for the AgentBrain.

### NPCAgent

The `NPCAgent` class is a subclass of `BaseAgent` and serves as the main class for users to interact with when creating chatbot personas. It has a single sub-brain, the 'extraction brain', which collects character sheet information from a Vector Store Index (`GPTSimpleVectorIndex`) at a directory, constructs a summary of that information, and adds it to the prompt.

#### Key Functions

- `__init__`: Initializes the NPCAgent instance with required parameters.
- `make_npc_agent`: Makes an NPC agent with character sheet and name.
- `load`: Asynchronously loads the NPC agent with the character sheet.

## How to Use

1. Import the necessary classes and modules.
2. Create an instance of the `NPCAgent` class.
3. Call the `load` method to asynchronously load the NPC agent with the character sheet.
4. Use the `async_query_master_brain` method to send queries to the Master Brain.

## Ideas for Extending

- Implement a web-based interface for creating and managing personas.
- Create a variety of pre-built personas for different contexts and industries.
- Integrate with other chat platforms beyond Discord.
- Develop a method for importing/exporting personas between users.
- Add additional tools and prompts for more diverse and dynamic conversations.

Further Resources:

Langchain:
- https://docs.langchain.com/docs/
- https://python.langchain.com/en/latest/index.html
- https://gpt-index.readthedocs.io/en/latest/index.html

Discord:
- https://discordpy.readthedocs.io/en/stable/api.html

Prompting:
- https://www.geoffreylitt.com/2023/02/26/llm-as-muse-not-oracle.html
- Prompting as programming language: https://arxiv.org/pdf/2212.06094.pdf
- Prompt Engineering    https://learnprompting.org/docs/advanced_applications/mrkl
- https://arxiv.org/pdf/2304.03442.pdf

=======
# ZaugBox
>>>>>>> origin/main
