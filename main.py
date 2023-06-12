from NPCFrameworkSDK.agents.npcagent import NPCAgent
import asyncio
from tuning_variables import *
from discordbot import NPCBot, BotRestartException
from NPCFrameworkSDK.modules.tools import Toolbelt
from NPCFrameworkSDK.modules.dataloader import IndexManager

from api import NPCPackage, SteamshipChatOpenAI, main
from langchain.callbacks.base import CallbackManager
from NPCFrameworkSDK.customlang.intermediatesteps_callbackmanager import CustomStreamingStdOutCallbackHandler


#This file runs our NPC Discord Bot with values from 'tuning_variables.py'
#create tools
async def runDiscordAgent(npcagent: NPCAgent, serverID: str):
  bot_restarted = False

  while True:

    # Create a bot instance
    bot = NPCBot(npcagent, serverID, npcagent.master_ai_prefix)

    if bot_restarted == True:
      bot.restart_channel_id = restart_channel_id
      bot.restart_requested = restart_requested
      bot_restarted = False

    bot_task = asyncio.create_task(bot.botmain(npcagent, serverID))

    # Wait for the bot to complete or the restart event to be set
    done, pending = await asyncio.wait(
      [bot_task, bot.restart_event.wait()],
      return_when=asyncio.FIRST_COMPLETED)

    # If the bot task is completed, break the loop
    if bot_task in done:
      break

    # If the restart event is set, restart the bot
    if bot.restart_event.is_set():
      print("Restarting the bot instance due to a restart request")
      restart_requested = bot.restart_requested
      restart_channel_id = bot.restart_channel_id
      print(f"Restart channel id: {restart_channel_id}"
            f" Restart requested: {restart_requested}")
      await bot.close()

      bot.agent.master_tools.index_manager.delete_index()

      # Rebuild cached saved files
      bot.character_files, bot.choices = NPCBot.rebuild_characterpersona_cache(
        bot.persona_directory)
      bot.brain, _ = await bot.load_agent_with_character_file(
        bot.persona_directory, default_persona_file_name, bot.agent)

      bot.restart_event.clear()

      bot_restarted = True
    #await asyncio.sleep(1)  # Wait for 1 second before retrying


asyncio.run(main())
#asyncio.run(runDiscordAgent(agent, serverID))
