from langchain.agents import ZeroShotAgent
from typing import Tuple, Sequence, Optional, Any, List
import re

FINAL_ANSWER_ACTION = "Final Answer:"


class CustomZeroShotAgent(ZeroShotAgent):

  def get_action_and_input(llm_output: str) -> Tuple[str, str]:
    """Parse out the action and input from the LLM output.
  
      Note: if you're specifying a custom prompt for the ZeroShotAgent,
      you will need to ensure that it meets the following Regex requirements.
      The string starting with "Action:" and the following string starting
      with "Action Input:" should be separated by a newline.
    """
    if FINAL_ANSWER_ACTION in llm_output:
      return "Final Answer", llm_output.split(FINAL_ANSWER_ACTION)[-1].strip()
    regex = r"Action: (.*?)[\n]*Action Input: (.*)"
    match = re.search(regex, llm_output, re.DOTALL)
    if not match:
      None
    action = match.group(1).strip()
    action_input = match.group(2)
    return action, action_input.strip(" ").strip('"')

  def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
    print("Here!")
    return self.get_action_and_input(text)
