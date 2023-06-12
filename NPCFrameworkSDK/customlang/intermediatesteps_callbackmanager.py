"""Callback Handler streams to stdout on new llm token."""
import sys
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.streamlit import StreamlitCallbackHandler
import streamlit as st


class CustomStreamingStdOutCallbackHandler(BaseCallbackHandler):
  """Callback handler for streaming. Only works with LLMs that support streaming."""

  def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str],
                   **kwargs: Any) -> None:
    """Run when LLM starts running."""

  def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
    """Run on new LLM token. Only available when streaming is enabled."""
    #print(token)

    sys.stdout.write(token)
    sys.stdout.flush()

  def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
    """Run when LLM ends running."""

  def on_llm_error(self, error: Union[Exception, KeyboardInterrupt],
                   **kwargs: Any) -> None:
    """Run when LLM errors."""

  def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any],
                     **kwargs: Any) -> None:
    """Run when chain starts running."""

  def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
    """Run when chain ends running."""

  def on_chain_error(self, error: Union[Exception, KeyboardInterrupt],
                     **kwargs: Any) -> None:
    """Run when chain errors."""

  def on_tool_start(self, serialized: Dict[str, Any], input_str: str,
                    **kwargs: Any) -> None:
    """Run when tool starts running."""

  def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
    """Run on agent action."""
    pass

  def on_tool_end(self, output: str, **kwargs: Any) -> None:
    """Run when tool ends running."""

  def on_tool_error(self, error: Union[Exception, KeyboardInterrupt],
                    **kwargs: Any) -> None:
    """Run when tool errors."""

  def on_text(self, text: str, **kwargs: Any) -> None:
    """Run on arbitrary text."""

  def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
    """Run on agent end."""


class CustomCallbackHandler(StdOutCallbackHandler):

  def on_new_token(self, token, run_id, parent_run_id):
    print(token)
    return token


class CustomStreamlitCallbackHandler(StreamlitCallbackHandler):

  def __init__(self, sessionstate: st.session_state) -> None:
    self.sessionstate = sessionstate
    self.tokens_stream = ""
    self.container = None

  def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
    """Run on new LLM token. Only available when streaming is enabled."""
    self.tokens_stream += token
    #self.tokens_area.text(self.tokens_stream)  # Write the entire string
    self.container.write(self.tokens_stream)

  def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
    """Do nothing."""
    #self.tokens_area.empty()
    self.tokens_stream = ""  # Clear the string
    self.container.write("")

  def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str],
                   **kwargs: Any) -> None:
    """Print out the prompts."""
    pass

  def on_llm_error(self, error: Union[Exception, KeyboardInterrupt],
                   **kwargs: Any) -> None:
    """Do nothing."""
    pass

  def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any],
                     **kwargs: Any) -> None:
    """Print out that we are entering a chain."""
    #class_name = serialized["name"]
    #st.write(f"Entering new {class_name} chain...")
    pass

  def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
    """Print out that we finished a chain."""
    st.write("Finished chain.")

  def on_chain_error(self, error: Union[Exception, KeyboardInterrupt],
                     **kwargs: Any) -> None:
    """Do nothing."""
    pass

  def on_tool_start(
    self,
    serialized: Dict[str, Any],
    input_str: str,
    **kwargs: Any,
  ) -> None:
    """Print out the log in specified color."""
    pass

  def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
    """Run on agent action."""
    # st.write requires two spaces before a newline to render it
    st.markdown(action.log.replace("\n", "  \n"))

  def on_tool_end(
    self,
    output: str,
    observation_prefix: Optional[str] = None,
    llm_prefix: Optional[str] = None,
    **kwargs: Any,
  ) -> None:
    """If not the final action, print out observation."""
    #st.write(f"{observation_prefix}{output}")
    #st.write(llm_prefix)
    pass

  def on_tool_error(self, error: Union[Exception, KeyboardInterrupt],
                    **kwargs: Any) -> None:
    """Do nothing."""
    pass

  def on_text(self, text: str, **kwargs: Any) -> None:
    """Run on text."""
    # st.write requires two spaces before a newline to render it
    #st.write(text.replace("\n", "  \n"))
    pass

  def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
    """Run on agent end."""
    # st.write requires two spaces before a newline to render it
    #st.write(finish.log.replace("\n", "  \n"))
    pass
