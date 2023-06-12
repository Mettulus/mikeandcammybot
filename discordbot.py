import discord
from discord.ext import commands
from discord import app_commands
from discord.ui import Button

from discord.ext import commands as CommandChoice
from NPCFrameworkSDK.agents.npcagent import NPCAgent
import os
import time
import asyncio
import functools
from discord.ext.commands.parameters import param
import typing
from tuning_variables import model_name, default_persona_file_name


class NPCBot(discord.Client):

  def __init__(self, agent: NPCAgent, guildID: str, botname: str):

    self.restart_event = asyncio.Event()  #init intents
    intents = discord.Intents.default()
    intents.message_content = True
    intents.messages = True
    intents.members = True
    intents.guilds = True
    intents.guild_messages = True
    intents.typing = True
    intents.guild_typing = True
    super().__init__(intents=intents)
    print(str(discord.Intents.message_content))

    self.botname = botname
    self.client = discord.client
    self.guildID = guildID
    self.synced = False
    self.queue = asyncio.Queue()

    self.restart_requested = False
    self.restart_channel_id = None

    self.agent = agent
    self.persona_directory = self.agent.persona_data_dir + "/persona_files"
    self.data_directory = self.agent.persona_data_dir + "/data"

    self.default_character_sheet_file_name = default_persona_file_name  #default_persona.txt
    self.character_files, self.choices = NPCBot.rebuild_characterpersona_cache(
      self.persona_directory)
    #intialize command tree to enable slash commands
    self.tree = app_commands.CommandTree(self)

    @self.tree.command(
      name="setcharacterpersona",
      description=
      "Set current Persona to file saved in the persona directory or generate a new one",
      guild=discord.Object(id=self.guildID))
    @commands.has_permissions(administrator=True)
    @app_commands.describe(option="Available Personas")
    @app_commands.choices(option=self.choices)
    async def load_character(interaction: discord.Interaction,
                             option: app_commands.Choice[str]):

      # Defer the response
      await interaction.response.defer()

      if option.value == "Generate New Persona":
        self.brain = await self.agent.load("", model_name)
        await interaction.followup.send(
          "New Character Persona Generated **(use 'savecharacterpersona' to save it)**"
        )
      else:

        self.brain, error_message = await NPCBot.load_agent_with_character_file(
          self.persona_directory, option.value, self.agent)

        if error_message:
          await interaction.followup.send(error_message)
        else:
          #await self.restart_bot()

          self.agent.reload_memory()
          await interaction.followup.send(f"Loaded Persona: {option.value}")

      #rebuild cached saved files

    @self.tree.command(name="savecharacterpersona",
                       description="Save current Persona to local drive",
                       guild=discord.Object(id=self.guildID))
    @commands.has_permissions(administrator=True)
    async def save_character(interaction: discord.Interaction, filename: str):

      charactersheet = self.agent.character_sheet
      dir = self.persona_directory
      #rebuild cached saved files
      self.save_character_file(dir, filename, charactersheet)
      await interaction.response.send_message(
        f"Persona Saved: {filename}. Bot restarting...")
      await self.restart_bot(interaction)

    @self.tree.command(
      name="uploadcharacterpersona",
      description=
      "Upload a Persona text file. These files are added to '(persona data folder)/persona_files'",
      guild=discord.Object(id=self.guildID))
    @commands.has_permissions(administrator=True)
    async def upload_character_persona(interaction: discord.Interaction,
                                       file: discord.Attachment):
      # Check if a file was uploaded
      if not file:
        await interaction.response.send_message(
          "Please upload a text file. Accepted file types: .txt, .rtf, .doc",
          ephemeral=True)
        return

      # If a string is passed, assume it's the filename and try to open the file
      if isinstance(file, str):
        try:
          with open(file, "rb") as f:
            file = discord.File(f)
        except Exception as e:
          await interaction.response.send_message(
            f"Failed to open file '{file}': {e}", ephemeral=True)
          return

      # Check if the file has a .txt extension
      if not file.filename.endswith(".txt") or file.filename.endswith(
          ".rtf") or file.filename.endswith(".doc"):
        await interaction.response.send_message(
          "Please upload a text file with a .txt, .rtf or .doc extension.",
          ephemeral=True)
        return

      # Read the contents of the file
      try:
        persona_data = await file.read()
        persona_data = persona_data.decode()  # Convert bytes to str

      except:
        await interaction.response.send_message(
          "Failed to read the file. Please try again.", ephemeral=True)
        return

      # Save the persona data to a file
      file_name = file.filename.split(".")[0]
      self.save_character_file(self.persona_directory, file_name, persona_data)

      # Send a confirmation message
      await interaction.response.send_message(
        f"Persona '{file_name}' uploaded and saved. Bot restarting...",
        ephemeral=True)

      # Rebuild cached saved files
      self.character_files, self.choices = NPCBot.rebuild_characterpersona_cache(
        self.persona_directory)

      await self.restart_bot(interaction)

    @self.tree.command(
      name="uploaddatafile",
      description=
      "Loaded into Long-Term Memory upon using 'loadcharacterpersona' or 'restartbot.'",
      guild=discord.Object(id=self.guildID))
    @commands.has_permissions(administrator=True)
    async def upload_file(interaction: discord.Interaction,
                          file: discord.Attachment):
      # Check if a file was uploaded
      if not file:
        await interaction.response.send_message("Please upload a file.",
                                                ephemeral=True)
        return

      # If a string is passed, assume it's the filename and try to open the file
      if isinstance(file, str):
        try:
          with open(file, "rb") as f:
            file = discord.File(f)
        except Exception as e:
          await interaction.response.send_message(
            f"Failed to open file '{file}': {e}", ephemeral=True)
          return

      # Read the contents of the file
      try:
        file_data = await file.read()
      except:
        await interaction.response.send_message(
          "Failed to read the file. Please try again.", ephemeral=True)
        return

      # Save the file data to the data directory
      file_name = file.filename
      file_path = os.path.join(self.data_directory, file_name)
      with open(file_path, "wb") as f:
        f.write(file_data)

      # Send a confirmation message
      await interaction.response.send_message(
        f"File '{file_name}' uploaded and saved to the data directory. Bot restarting...",
        ephemeral=True)
      await self.restart_bot(interaction)

    @self.tree.command(name="say",
                       description="What happens next?",
                       guild=discord.Object(id=self.guildID))
    async def say(interaction: discord.Interaction, message: str):

      channel = interaction.channel
      ephemeral = True

      #if max que size return
      if self.queue.qsize() >= 2:
        await interaction.response.send_message(
          content=
          f"Sorry, there are currently too many responses in the queue: {self.queue.qsize()}/2",
          ephemeral=True)
        return

      lines = message.split('\n')
      for line in lines:
        await self.queue.put((interaction.user, line))
      intermediate_steps = ""
      try:
        async with interaction.channel.typing():

          while True:
            try:

              while not self.queue.empty():
                user, query = await self.queue.get()

                user_query = f"**{user.name}**: {query}"
                finalquery = f"{user.name}: {query}".strip()

                mes = await channel.send(user_query)
                retry_count = 0

                await interaction.response.send_message(content="thinking...",
                                                        ephemeral=ephemeral,
                                                        delete_after=20)

              response, intermediate_steps = await self.agent.master_brain.async_query(
                finalquery)

              #except Exception as e:
              async def send_intermediate_steps():
                if intermediate_steps:
                  intermediate_steps.strip()
                  await mes.edit(content=f"\n\n*{intermediate_steps}*\n\n")

              await asyncio.create_task(send_intermediate_steps())
              await asyncio.sleep(1)
              if response:
                #Format intermediate steps output
                response.strip()
                response = f"*`{intermediate_steps}`*\n\n**{botname}**: " + response

                var2 = user_query + "\n\n" + response + "\n\n"

              else:
                return

              await mes.edit(content=var2)
              self.queue.task_done()
              break
            except discord.errors.HTTPException as e:
              if e.response.status == 429:
                # Use exponential backoff for rate limiting errors
                retry_after = 1
                wait_time = 2**retry_count * retry_after
                max_wait = 10
                retry_count += 1
                if wait_time > max_wait:
                  # Maximum wait time of 120 seconds
                  wait_time = max_wait
                await asyncio.sleep(wait_time)
      except discord.NotFound:
        # interaction already acknowledged or expired
        return

    @self.tree.command(name="restartbot",
                       description="Restart the bot instance",
                       guild=discord.Object(id=self.guildID))
    @commands.has_permissions(administrator=True)
    async def restart_bot_command(interaction: discord.Interaction):

      await interaction.response.send_message("Bot restarting...",
                                              ephemeral=True)

      await self.restart_bot(interaction)

  async def restart_bot(self, interaction: discord.Interaction):
    # Set the restart flag and store the channel ID
    self.restart_requested = True
    self.restart_channel_id = interaction.channel_id
    self.agent.reload_memory()
    self.restart_event.set()

  async def botmain(self, npcagent: NPCAgent, guildID: str):
    self.brain, _ = await NPCBot.load_agent_with_character_file(
      self.persona_directory, self.default_character_sheet_file_name, npcagent)
    my_secret = os.environ['DISCORD_BOT_SECRET']
    await self.start(my_secret)

  @staticmethod
  def get_character_files_array(data_folder: str):
    character_files = []
    for file in os.listdir(data_folder):
      if file.endswith((".txt", ".rtf", ".doc")):
        character_files.append(file)
    return character_files

  @staticmethod
  def load_character_file(data_folder: str, filename: str):
    file_path = os.path.join(data_folder, filename)
    with open(file_path, "r") as file:
      character_data = file.read()
    return character_data

  @staticmethod
  def save_character_file(data_folder: str, filename: str,
                          character_data: str):
    # Create the directory if it doesn't exist
    if not os.path.exists(data_folder):
      os.makedirs(data_folder)

    # Set a file name
    file_name = f"{filename}.txt"

    # Combine the directory path and the file name
    file_path = os.path.join(data_folder, file_name)

    # Write the contents of the charactersheet to the file
    with open(file_path, 'w') as file:
      file.write(character_data)

    # Send a message saying "Character Saved" in Discord

  @staticmethod
  async def load_agent_with_character_file(data_folder: str, filename: str,
                                           agent: NPCAgent):
    try:
      default_character_data = NPCBot.load_character_file(
        data_folder, filename)
    except FileNotFoundError:
      return None, f"File '{filename}' not found in the data folder '{data_folder}'. Please check the file name and try again."

    brain = await agent.load(default_character_data, model_name)
    return brain, None

  @staticmethod
  def rebuild_characterpersona_cache(data_folder: str):
    character_files = NPCBot.get_character_files_array(data_folder)
    slash_command_choices = [
      app_commands.Choice(name="Generate New Persona",
                          value="Generate New Persona"),
      *[
        app_commands.Choice(name=file, value=file) for file in character_files
      ],
    ]

    return character_files, slash_command_choices

  async def on_ready(self):

    print("start")
    await self.wait_until_ready()
    if not self.synced:
      await self.tree.sync(guild=discord.Object(id=self.guildID))
      self.synced = True

    print(f"I'm in {self.user}")

    if self.restart_requested:
      self.restart_requested = False
      channel = self.get_channel(self.restart_channel_id)
      if channel:
        await channel.send("Bot restarted")


class BotRestartException(Exception):
  pass
