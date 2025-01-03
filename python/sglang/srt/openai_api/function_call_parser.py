import json
import re
from abc import ABC, abstractmethod
from json import JSONDecodeError, JSONDecoder
from typing import Any, Dict, List, Optional, Tuple

import partial_json_parser
from partial_json_parser.core.options import Allow

from sglang.srt.openai_api.protocol import Tool, ToolCallItem


def _find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def _partial_json_loads(input_str: str, flags: Allow) -> Tuple[Any, int]:
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except JSONDecodeError as e:
        if "Extra data" in e.msg:
            dec = JSONDecoder()
            return dec.raw_decode(input_str)
        raise


def _is_complete_json(input_str: str) -> bool:
    try:
        json.loads(input_str)
        return True
    except JSONDecodeError:
        return False


def _find_common_suffix(s1: str, s2: str) -> str:
    """
    Finds a common suffix shared between two strings, if there is one. Order of
    arguments is NOT important.
    Stops when the suffix ends OR it hits an alphanumeric character

    e.g. find_common_suffix('{"fruit": "ap"}', '{"fruit": "apple"}') -> '"}'
    """
    suffix = ""
    min_length = min(len(s1), len(s2))
    for i in range(1, min_length + 1):
        if s1[-i] == s2[-i] and not s1[-i].isalnum():
            suffix = s1[-i] + suffix
        else:
            break
    return suffix


def _extract_intermediate_diff(curr: str, old: str) -> str:
    """
    Given two strings, extract the difference in the middle between two strings
    that are known to have a common prefix and/or suffix.

    This function is provided as a UTILITY for extracting information from JSON
    generated by partial_json_parser, to help in ensuring that the right tokens
    are returned in streaming, so that close-quotes, close-brackets and
    close-braces are not returned prematurely. The order of arguments IS
    important - the new version of the partially-parsed JSON must be the first
    argument, and the secnod argument must be from the previous generation.

    What it returns, is tokens that should be streamed to the client.

    e.g. extract_intermediate_diff('{"fruit": "apple"}', '{"fruit": "ap"}')
        -> 'ple'

    """
    suffix = _find_common_suffix(curr, old)

    old = old[::-1].replace(suffix[::-1], "", 1)[::-1]
    prefix = _find_common_prefix(curr, old)
    diff = curr
    if len(suffix):
        diff = diff[::-1].replace(suffix[::-1], "", 1)[::-1]

    if len(prefix):
        # replace the prefix only once in case it's mirrored
        diff = diff.replace(prefix, "", 1)

    return diff


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(
        self, normal_text: str = "", calls: Optional[List[ToolCallItem]] = None
    ):
        self.normal_text = normal_text
        self.calls = calls or []


class BaseFormatDetector(ABC):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    @abstractmethod
    def detect_and_parse(self, text: str, tools: List[Tool]) -> List[ToolCallItem]:
        """
        Parses the text in one go. Returns success=True if the format matches, otherwise False.
        Note that leftover_text here represents "content that this parser will not consume further".
        """
        pass

    @abstractmethod
    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing, internally maintains a buffer or state.
        Each time new_text is received, attempts to parse it. If a function call is matched, returns calls.
        It may also return some normal_text.
        """
        pass


class Qwen25Detector(BaseFormatDetector):
    """
    Detector for Qwen 2.5 models.
    Assumes function call format:
      <tool_call>{"name":"xxx", "arguments":{...}}</tool_call>
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self._buffer = ""
        self.bot_token = "<tool_call>"

        # Indicates the index of the tool call currently being processed; -1 means not started or already finished
        self.current_tool_id: int = -1
        # Indicates whether the name of the current tool has already been output (it will only be output once for the same function call)
        self.current_tool_name_sent: bool = False
        # Stores the arguments (strings) already sent for each tool, for incremental sending
        self.streamed_args_for_tool: List[str] = []
        # Stores the list of all tool calls (JSON objects) parsed in the "previous" iteration
        self.prev_tool_call_arr: List[Dict] = []

        self.tool_call_regex = re.compile(r"\[{.*?}\]", re.DOTALL)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> List[ToolCallItem]:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        # breakpoint()
        if "<tool_call>" not in text:
            return []
        pattern = r"<tool_call>(.*?)</tool_call>"
        match_result_list = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match_result in match_result_list:
            action = json.loads(match_result)
            name, parameters = action["name"], json.dumps(
                action.get("parameters", action.get("arguments", {})),
                ensure_ascii=False,
            )
            tool_index = [tool.function.name for tool in tools].index(name)
            tool_call_item = ToolCallItem(
                tool_index=tool_index, name=name, parameters=parameters
            )
            calls.append(tool_call_item)
        return calls

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing, referencing the logic of Llama32Detector.
        We partially parse JSON within <tool_call>...</tool_call>, and handle
        incremental argument output.
        """
        # Append new text to buffer
        self._buffer += new_text
        current_text = self._buffer
        if not (
            current_text.startswith(self.bot_token) or current_text.startswith("{")
        ):
            return StreamingParseResult(normal_text=new_text)

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        # breakpoint()
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = []
            is_complete = []
            try:
                # depending on the prompt format the Llama model may or may not
                # prefix the output with the <|python_tag|> token
                start_idx = (
                    len(self.bot_token)
                    if current_text.startswith(self.bot_token)
                    else 0
                )
                while start_idx < len(current_text):
                    (obj, end_idx) = _partial_json_loads(
                        current_text[start_idx:], flags
                    )
                    is_complete.append(
                        _is_complete_json(current_text[start_idx : start_idx + end_idx])
                    )
                    start_idx += end_idx + len("; ")
                    # depending on the prompt Llama can use
                    # either arguments or parameters
                    if "parameters" in obj:
                        assert (
                            "arguments" not in obj
                        ), "model generated both parameters and arguments"
                        obj["arguments"] = obj["parameters"]
                    tool_call_arr.append(obj)

            except partial_json_parser.core.exceptions.MalformedJSON:
                print("not enough tokens to parse into JSON yet")
                return StreamingParseResult()

            # select as the current tool call the one we're on the state at
            current_tool_call: Dict = (
                tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            )

            # case -- if no tokens have been streamed for the tool, e.g.
            #   only the array brackets, stream nothing
            if len(tool_call_arr) == 0:
                return StreamingParseResult()

            # case: we are starting a new tool in the array
            #   -> array has > 0 length AND length has moved past cursor
            elif (
                len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1
            ):

                # if we're moving on to a new call, first make sure we
                # haven't missed anything in the previous one that was
                # auto-generated due to JSON completions, but wasn't
                # streamed to the client yet.
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments)
                        sent = len(self.streamed_args_for_tool[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]

                        print("got arguments diff: %s", argument_diff)
                        res = StreamingParseResult(
                            normal_text=None,
                            calls=[
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name="",
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += argument_diff
                    else:
                        res = StreamingParseResult()
                else:
                    res = StreamingParseResult()
                # re-set stuff pertaining to progress in the current tool
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                print("starting on new tool %d", self.current_tool_id)
                return res

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            elif not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    res = StreamingParseResult(
                        normal_text=None,
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        ],
                    )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            # now we know we're on the same tool call and we're streaming
            # arguments
            else:
                cur_arguments = current_tool_call.get("arguments")
                res = StreamingParseResult()

                if cur_arguments:
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments)
                    prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                        "arguments"
                    )

                    argument_diff = None
                    if is_complete[self.current_tool_id]:
                        argument_diff = cur_args_json[sent:]
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments)
                        if cur_args_json != prev_args_json:

                            prefix = _find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    if argument_diff is not None:
                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name="",
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += argument_diff

            self.prev_tool_call_arr = tool_call_arr
            return res

        except Exception:
            print("Error trying to handle streaming tool call.")
            print("Skipping chunk as a result of tool streaming extraction " "error")
            return StreamingParseResult()


class MistralDetector(BaseFormatDetector):
    """
    Detector for Mistral models.
    Assumes function call format:
      <|action_start|><|plugin|>{"name":"xxx", "arguments":{...}}<|action_end|>
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        # initialize properties used for state when parsing tool calls in
        self.buffer = ""
        # streaming mode
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: List[str] = (
            []
        )  # map what has been streamed for each tool so far to a list
        self.bot_token = "[TOOL_CALLS]"
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)

    def _clean_text(self, text: str) -> str:
        """
        clean text to only leave ''[TOOL_CALLS] [{"name": xxx, "arguments": {xxx}}]'
        for example,
            text = '[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}]\n\nToday\'s weather in Boston is :{function call result} (in Fahrenheit)\n\nIf you prefer Celsius, please let me know.'
            return '[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}]'
        The key pattern is [TOOL_CALLS] [...]
        """
        return re.findall(r"\[TOOL_CALLS\] \[.*?\]", text, re.DOTALL)[0]

    def detect_and_parse(self, text: str, tools: List[Tool]) -> List[ToolCallItem]:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        text = self._clean_text(text)
        tool_content = text.replace(self.bot_token, "").strip()
        raw_tool_call = self.tool_call_regex.findall(tool_content)[0]
        function_call_arr = json.loads(raw_tool_call)
        calls = []
        for match_result in function_call_arr:
            action = match_result
            name, parameters = action["name"], json.dumps(
                action.get("parameters", action.get("arguments", {})),
                ensure_ascii=False,
            )
            tool_index = [tool.function.name for tool in tools].index(name)
            tool_call_item = ToolCallItem(
                tool_index=tool_index, name=name, parameters=parameters
            )
            calls.append(tool_call_item)
        return calls

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:

        delta_text = new_text
        self.buffer += delta_text
        current_text = self.buffer

        # if the tool call token is not in the tokens generated so far, append
        # output to contents since it's not a tool
        if self.bot_token not in current_text:
            return StreamingParseResult(normal_text=delta_text)

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:

            # replace BOT token with empty string, and convert single quotes
            # to double to allow parsing as JSON since mistral uses single
            # quotes instead of double for tool calls
            parsable_arr = current_text.split(self.bot_token)[-1]

            # tool calls are generated in an array, so do partial JSON
            # parsing on the entire array
            try:
                tool_call_arr: List[Dict] = partial_json_parser.loads(
                    parsable_arr, flags
                )
            except partial_json_parser.core.exceptions.MalformedJSON:
                print("not enough tokens to parse into JSON yet")
                return StreamingParseResult()

            # select as the current tool call the one we're on the state at

            current_tool_call: Dict = (
                tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            )

            # case -- if no tokens have been streamed for the tool, e.g.
            #   only the array brackets, stream nothing
            if len(tool_call_arr) == 0:
                return StreamingParseResult()

            # case: we are starting a new tool in the array
            #   -> array has > 0 length AND length has moved past cursor
            elif (
                len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1
            ):

                # if we're moving on to a new call, first make sure we
                # haven't missed anything in the previous one that was
                # auto-generated due to JSON completions, but wasn't
                # streamed to the client yet.
                if self.current_tool_id >= 0:
                    diff = current_tool_call.get("arguments")

                    if diff:
                        diff = json.dumps(diff, ensure_ascii=False).replace(
                            self.streamed_args_for_tool[self.current_tool_id], ""
                        )
                        delta = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name="",
                                    parameters=diff,
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += diff
                    else:
                        delta = StreamingParseResult()
                else:
                    delta = StreamingParseResult()
                # re-set stuff pertaining to progress in the current tool
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                print("starting on new tool %d", self.current_tool_id)
                return delta

            # case: update an existing tool - this is handled below

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:

                    delta = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        ]
                    )
                    self.current_tool_name_sent = True
                else:
                    delta = StreamingParseResult()

            # now we know we're on the same tool call and we're streaming
            # arguments
            else:

                prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                    "arguments"
                )
                cur_arguments = current_tool_call.get("arguments")

                new_text = delta_text.replace("'", '"')
                if '"}' in new_text:
                    new_text = new_text[: new_text.rindex('"}')]

                if not cur_arguments and not prev_arguments:

                    delta = StreamingParseResult()
                elif not cur_arguments and prev_arguments:
                    print(
                        "INVARIANT - impossible to have arguments reset "
                        "mid-arguments"
                    )
                    delta = StreamingParseResult()
                elif cur_arguments and not prev_arguments:
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)[
                        :-2
                    ]
                    print("finding %s in %s", new_text, cur_arguments_json)

                    if new_text not in cur_arguments_json:
                        return StreamingParseResult()
                    arguments_delta = cur_arguments_json[
                        : cur_arguments_json.rindex(new_text) + len(new_text)
                    ]
                    print("First tokens in arguments received: %s", arguments_delta)
                    delta = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name="",
                                parameters=arguments_delta,
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += arguments_delta

                elif cur_arguments and prev_arguments:
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                    print(
                        "Searching for diff between \n%s\n%s",
                        cur_args_json,
                        prev_args_json,
                    )

                    argument_diff = _extract_intermediate_diff(
                        cur_args_json, prev_args_json
                    )
                    print("got arguments diff: %s", argument_diff)
                    delta = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name="",
                                parameters=argument_diff,
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff
                else:
                    # try parsing it with regular JSON - if it works we're
                    # at the end, and we need to send the difference between
                    # tokens streamed so far and the valid JSON
                    delta = StreamingParseResult()

            # check to see if the name is defined and has been sent. if so,
            # stream the name - otherwise keep waiting
            # finish by setting old and returning None as base case
            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception:
            print("Error trying to handle streaming tool call.")
            print("Skipping chunk as a result of tool streaming extraction " "error")
            return StreamingParseResult()


class Llama32Detector(BaseFormatDetector):
    """
    Detector for Llama 3.2 models.
    Assumes function call format:
      <|python_tag|>{"name":"xxx", "arguments":{...}}
    Does not require a closing tag "</python_tag|>",
    relies on json.loads(...) success to determine if JSON is complete.
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self._buffer = ""
        self.bot_token = "<|python_tag|>"

        # Indicates the index of the tool call currently being processed; -1 means not started or already finished
        self.current_tool_id: int = -1
        # Indicates whether the name of the current tool has already been output (it will only be output once for the same function call)
        self.current_tool_name_sent: bool = False
        # Stores the arguments (strings) already sent for each tool, for incremental sending
        self.streamed_args_for_tool: List[str] = []
        # Stores the list of all tool calls (JSON objects) parsed in the "previous" iteration
        self.prev_tool_call_arr: List[Dict] = []

        self.tool_call_regex = re.compile(r"\[{.*?}\]", re.DOTALL)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> List[ToolCallItem]:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """

        if "<|python_tag|>" not in text:
            return []
        _, action = text.split("<|python_tag|>")
        action = json.loads(action)
        name, parameters = action["name"], json.dumps(
            action.get("parameters", action.get("arguments", {})),
            ensure_ascii=False,
        )
        tool_index = [tool.function.name for tool in tools].index(name)
        tool_call_item = ToolCallItem(
            tool_index=tool_index, name=name, parameters=parameters
        )
        calls = [tool_call_item]
        return calls

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        current_text = self._buffer
        if not (
            current_text.startswith(self.bot_token) or current_text.startswith("{")
        ):
            return StreamingParseResult(normal_text=new_text)

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        # breakpoint()
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = []
            is_complete = []
            try:
                # depending on the prompt format the Llama model may or may not
                # prefix the output with the <|python_tag|> token
                start_idx = (
                    len(self.bot_token)
                    if current_text.startswith(self.bot_token)
                    else 0
                )
                while start_idx < len(current_text):
                    (obj, end_idx) = _partial_json_loads(
                        current_text[start_idx:], flags
                    )
                    is_complete.append(
                        _is_complete_json(current_text[start_idx : start_idx + end_idx])
                    )
                    start_idx += end_idx + len("; ")
                    # depending on the prompt Llama can use
                    # either arguments or parameters
                    if "parameters" in obj:
                        assert (
                            "arguments" not in obj
                        ), "model generated both parameters and arguments"
                        obj["arguments"] = obj["parameters"]
                    tool_call_arr.append(obj)

            except partial_json_parser.core.exceptions.MalformedJSON:
                print("not enough tokens to parse into JSON yet")
                return StreamingParseResult()

            # select as the current tool call the one we're on the state at
            current_tool_call: Dict = (
                tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            )

            # case -- if no tokens have been streamed for the tool, e.g.
            #   only the array brackets, stream nothing
            if len(tool_call_arr) == 0:
                return StreamingParseResult()

            # case: we are starting a new tool in the array
            #   -> array has > 0 length AND length has moved past cursor
            elif (
                len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1
            ):

                # if we're moving on to a new call, first make sure we
                # haven't missed anything in the previous one that was
                # auto-generated due to JSON completions, but wasn't
                # streamed to the client yet.
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments)
                        sent = len(self.streamed_args_for_tool[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]

                        print("got arguments diff: %s", argument_diff)
                        res = StreamingParseResult(
                            normal_text=None,
                            calls=[
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name="",
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += argument_diff
                    else:
                        res = StreamingParseResult()
                else:
                    res = StreamingParseResult()
                # re-set stuff pertaining to progress in the current tool
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                print("starting on new tool %d", self.current_tool_id)
                return res

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            elif not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    res = StreamingParseResult(
                        normal_text=None,
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        ],
                    )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            # now we know we're on the same tool call and we're streaming
            # arguments
            else:
                cur_arguments = current_tool_call.get("arguments")
                res = StreamingParseResult()

                if cur_arguments:
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments)
                    prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                        "arguments"
                    )

                    argument_diff = None
                    if is_complete[self.current_tool_id]:
                        argument_diff = cur_args_json[sent:]
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments)
                        if cur_args_json != prev_args_json:

                            prefix = _find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    if argument_diff is not None:
                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name="",
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += argument_diff

            self.prev_tool_call_arr = tool_call_arr
            return res

        except Exception:
            print("Error trying to handle streaming tool call.")
            print("Skipping chunk as a result of tool streaming extraction " "error")
            return StreamingParseResult()


class MultiFormatParser:
    def __init__(self, detectors: List[BaseFormatDetector]):
        """
        :param detectors: A series of available Detector instances passed in
        """
        self.detectors = detectors

    def parse_once(self, text: str, tools: List[Tool]):
        """
        One-time parsing: Loop through detectors until there are no new matches or text is exhausted
        Return: (final_text, all_calls)
        - final_text: The remaining text after parsing that was not consumed by any Detector (can be treated as normal text)
        - all_calls: All calls parsed by the Detectors
        """
        final_calls = []
        final_normal_text = text
        for detector in self.detectors:
            tool_call_list = detector.detect_and_parse(text, tools)
            final_calls.extend(tool_call_list)

        # leftover_text is the normal text not consumed by any Detector
        return final_normal_text, final_calls

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]):
        """
        Streaming incremental parsing: Feed new_text to each detector's parse_streaming_increment
        and merge their produced normal_text/calls to return.
        (The logic here can be "priority-based" or "parallel parsing" based on your needs)
        """
        final_normal_text = ""
        final_calls = []

        for detector in self.detectors:
            sp_result = detector.parse_streaming_increment(new_text, tools)
            # Merge normal_text and calls
            # If one sp_result contains result call, this should be a successful parse
            # If one sp_result only contains normal_text, this can either be a successful
            # parse or it is not using the desired parsing tool.
            if sp_result.normal_text:
                final_normal_text = sp_result.normal_text
            if sp_result.calls:
                final_calls.extend(sp_result.calls)
                final_normal_text = sp_result.normal_text
                break

        return final_normal_text, final_calls


class FunctionCallParser:
    """
    In streaming scenarios, each time new_text is received, it calls multi_format_parser.parse_streaming_increment
    and returns the resulting normal_text and calls to the upper layer (or SSE).
    """

    def __init__(self, tools: List[Tool]):
        # Inject a set of Detectors here. To support Qwen25, InternLM2 in the future,
        # simply instantiate the corresponding Detector and add it to the list:
        self.multi_format_parser = MultiFormatParser(
            [
                Llama32Detector(),
                # Qwen25Detector(),
                # MistralDetector(),
                # ...
            ]
        )
        self.tools = tools

    def parse_non_stream(self, full_text: str):
        """
        Non-streaming call: one-time parsing
        """
        full_normal_text, calls = self.multi_format_parser.parse_once(
            full_text, self.tools
        )
        return full_normal_text, calls

    def parse_stream_chunk(self, chunk_text: str):
        """
        Streaming call: incremental parsing
        """
        normal_text, calls = self.multi_format_parser.parse_streaming_increment(
            chunk_text, self.tools
        )
        return normal_text, calls
