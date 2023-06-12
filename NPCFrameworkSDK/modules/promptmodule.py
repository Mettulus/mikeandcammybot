from langchain.prompts import PromptTemplate
from NPCFrameworkSDK.customlang.customzeroshotagent import CustomZeroShotAgent
from langchain.agents import Tool, tool
from typing import Dict, List, Any, Tuple
from tuning_variables import persona_extraction_formatinstructions, persona_extraction_prefix, persona_extraction_suffix, entity_extraction_prefix, entity_extraction_suffix, npc_prefix, npc_suffix, ChatNPC_Format_Instructions


class PromptModule:
  """Data class for prompts to be passed into Brains. 
  
  Prompts are `Prompt Template` objects made to be passed into the 'Agent Brain' class, which takes in a prefix and suffix to construct a prompt with access to passed in tools"""

  def __init__(self):
    """Format Instructions that get passed to prompts. Tell the Langchain agents how to chain together thoughts"""

  # Method to create a prompt for character sheet extraction
  def persona_extraction_prompt(
      self, toolkit: List[Tool]) -> Tuple[PromptTemplate, Dict[str, Any]]:
    """Prompt used for extracting Character Sheet information from a given index.
    Returns a `PromptTemplate`object
    """
    input_variables = ["name", "chat_history", 'input', 'agent_scratchpad']

    prefix = persona_extraction_prefix
    suffix = persona_extraction_suffix
    formatinstructions = persona_extraction_formatinstructions
    prompt, input_dict = self.construct_zero_shot_prompt_template(
      prefix, suffix, toolkit, formatinstructions, input_variables)

    input_dict["chat_history"] = []
    return prompt, input_dict

  # Method to create a prompt for character sheet extraction
  def entity_extraction_prompt(
      self, toolkit: List[Tool]) -> Tuple[PromptTemplate, Dict[str, Any]]:
    """Prompt used for extracting Character Sheet information from a given index.
    
    """

    #    Non Chat Prompt so must include default input vriables

    input_variables = ["name", "chat_history", 'input', 'agent_scratchpad']

    prefix = entity_extraction_prefix
    suffix = entity_extraction_suffix
    formatinstructions = persona_extraction_formatinstructions
    prompt, input_dict = self.construct_zero_shot_prompt_template(
      prefix, suffix, toolkit, formatinstructions, input_variables)

    input_dict["chat_history"] = []
    return prompt, input_dict

  # Method to create a prompt for a chat NPC
  def chatnpc_prompt(
      self, toolkit: List[Tool]) -> Tuple[PromptTemplate, Dict[str, Any]]:
    """Prompt for the Master Brain of an 'NPC'

    Returns a `PromptTemplate` and a `dict` of input variables.

    input variables: "input", "chat_history", "agent_scratchpad", "name", "metaprompt"
    
    format instructions: NPCformatinstructions

    """

    input_variables = ["name", "metaprompt"]

    prefix = npc_prefix
    suffix = npc_suffix
    formatinstructions = ChatNPC_Format_Instructions
    prompt, input_dict = self.construct_zero_shot_prompt_template(
      prefix, suffix, toolkit, formatinstructions, input_variables)

    input_dict["chat_history"] = []

    print("ChatNPC")
    print(input_dict)

    return prompt, input_dict

  # Method to construct a zero-shot prompt template from a string suffix and prefix
  def construct_zero_shot_prompt_template(
      self, prefix: str, suffix: str, toolkit: List[Tool],
      format_instructions: str,
      input_variables: List[str]) -> Tuple[PromptTemplate, Dict[str, Any]]:
    """Directly create a Prompt Template from a string prefix & suffix"""

    input_dict = create_input_dict(input_variables)
    _prompt = CustomZeroShotAgent.create_prompt(
      format_instructions=format_instructions,
      tools=toolkit,
      prefix=prefix,
      suffix=suffix,
      input_variables=input_variables)

    return _prompt, input_dict


def create_input_dict(input_variables: List[str]) -> Dict[str, Any]:
  d = {variable: "" for variable in input_variables}
  d.update({"input": "", "chat_history": [], "agent_scratchpad": ""})
  return d


def dict_to_list(input_dict: Dict[str, Any]) -> List[str]:
  return list(input_dict.keys())
