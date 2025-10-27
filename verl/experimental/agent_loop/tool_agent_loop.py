# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import copy
import json
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        cls.overlong_filter = config.actor_rollout_ref.rollout.multi_turn.overlong_filter
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")
        
        # Load speaker mapping for RAG context
        cls._speaker_mapping = cls._load_speaker_mapping()

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

    @classmethod
    def _load_speaker_mapping(cls) -> dict:
        """Load speaker mapping from locomo10.json."""
        import json
        import os

        try:
            locomo_path = "/workspace/memupdate/data/locomo10.json"
            if not os.path.exists(locomo_path):
                print(f"⚠️ LoCoMo data not found at {locomo_path}, RAG context will be disabled")
                return {}

            with open(locomo_path, 'r') as f:
                locomo_data = json.load(f)

            speaker_mapping = {}
            for conv in locomo_data:
                sample_id = conv.get("sample_id", "")
                if sample_id and "conversation" in conv:
                    speaker_a = conv["conversation"].get("speaker_a", "")
                    speaker_b = conv["conversation"].get("speaker_b", "")
                    if speaker_a and speaker_b:
                        speaker_mapping[sample_id] = (speaker_a, speaker_b)

            print(f"✅ Loaded speaker mapping for {len(speaker_mapping)} conversations")
            return speaker_mapping

        except Exception as e:
            print(f"⚠️ Failed to load speaker mapping: {e}")
            return {}

    def _group_memories_by_context(self, memories: list[dict]) -> list[dict]:
        """Group memories by their context relationships for chunked formatting."""
        # First, identify all search results and create groups for them
        memory_groups = []
        result_memories = [m for m in memories if not m.get("is_context", False) or m.get("position") == "search_result"]
        
        # If no explicit results found (all are context or no position set), treat all non-context as results
        if not result_memories:
            result_memories = [m for m in memories if not m.get("is_context", False)]
        
        # Create a group for each search result with its associated context
        for result_mem in result_memories:
            result_id = result_mem.get("id")
            
            # Find all context memories that belong to this result using context_for field
            group = {
                "previous": [m for m in memories 
                            if m.get("is_context", False) and 
                               m.get("position") == "previous" and 
                               m.get("context_for") == result_id],
                "result": result_mem,
                "next": [m for m in memories 
                        if m.get("is_context", False) and 
                           m.get("position") == "next" and 
                           m.get("context_for") == result_id]
            }
            memory_groups.append(group)
        
        return memory_groups

    def _format_memory_metadata(self, metadata: dict) -> str:
        """Format metadata for consistent display in memory formatting."""
        metadata_parts = []
        if metadata.get("speaker"):
            metadata_parts.append(f"Speaker: {metadata['speaker']}")
        if metadata.get("source"):
            metadata_parts.append(f"Source: {metadata['source']}")
        if metadata.get("evidence"):
            metadata_parts.append(f"Evidence: {metadata['evidence']}")
        if metadata.get("session"):
            metadata_parts.append(f"Session: {metadata['session']}")
        if metadata.get("timestamp"):
            metadata_parts.append(f"Time: {metadata['timestamp']}")
        return f" [{' | '.join(metadata_parts)}]" if metadata_parts else ""

    async def _retrieve_initial_context(self, target_question: str, sample_id: str, trial_namespace: str) -> str:
        """Retrieve initial RAG context from conversation memories for both speakers."""
        try:
            # Look up speakers for this sample
            speakers = self._speaker_mapping.get(sample_id, (None, None))
            speaker_a, speaker_b = speakers

            if not speaker_a or not speaker_b:
                return ""
            
            from memupdate.tools.base_memory_tool import MemoryStoreManager
            
            # Run parallel queries for both speakers
            tasks = []
            if speaker_a:
                tasks.append(MemoryStoreManager.search_memory_via_actor_async(
                    trial_namespace=trial_namespace,
                    query=target_question,
                    limit=5,
                    source_filter="conversation",
                    speaker_filter=speaker_a,
                    search_type="semantic_search",
                    n_prev=2,
                    n_next=2
                ))
            if speaker_b:
                tasks.append(MemoryStoreManager.search_memory_via_actor_async(
                    trial_namespace=trial_namespace,
                    query=target_question,
                    limit=5,
                    source_filter="conversation", 
                    speaker_filter=speaker_b,
                    search_type="semantic_search",
                    n_prev=2,
                    n_next=2
                ))
            
            if not tasks:
                return ""
                
            # Execute queries in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Build rich context string with memory groups and metadata
            formatted_sections = []
            memory_counter = 1
            
            # Process results for both speakers
            speakers_data = [
                (speaker_a, results[0] if len(results) > 0 and not isinstance(results[0], Exception) else None),
                (speaker_b, results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None)
            ]
            
            for speaker, speaker_results in speakers_data:
                if not speaker or not speaker_results:
                    continue
                    
                if speaker_results.get("success") and speaker_results.get("results"):
                    memories = speaker_results["results"]
                    if not memories:
                        continue
                    
                    # Add section header with rich formatting
                    section_header = f"\n{'='*60}\nConversation Memories - {speaker} ({len(memories)} memories)\n{'='*60}"
                    section_lines = [section_header]
                    
                    # Group memories by context relationships
                    memory_groups = self._group_memories_by_context(memories)
                    
                    for group_idx, group in enumerate(memory_groups):
                        if group_idx > 0:
                            section_lines.append("")  # Blank line between groups within section
                        
                        # Add memory group header
                        section_lines.append(f"--- Memory Group {group_idx + 1} ---")
                        
                        # Format previous context
                        for prev_mem in group["previous"]:
                            metadata_str = self._format_memory_metadata(prev_mem.get("metadata", {}))
                            section_lines.append(f"Memory {memory_counter} [CONTEXT]{metadata_str}: {prev_mem.get('content', '')}")
                            memory_counter += 1
                        
                        # Format main result
                        if group["result"]:
                            result_mem = group["result"]
                            metadata_str = self._format_memory_metadata(result_mem.get("metadata", {}))
                            section_lines.append(f"Memory {memory_counter}{metadata_str}: {result_mem.get('content', '')}")
                            memory_counter += 1
                        
                        # Format next context
                        for next_mem in group["next"]:
                            metadata_str = self._format_memory_metadata(next_mem.get("metadata", {}))
                            section_lines.append(f"Memory {memory_counter} [CONTEXT]{metadata_str}: {next_mem.get('content', '')}")
                            memory_counter += 1
                    
                    formatted_sections.append("\n".join(section_lines))
            
            if formatted_sections:
                return "\n".join(formatted_sections)
            else:
                return ""
                
        except Exception as e:
            print(f"⚠️ Failed to retrieve initial context: {e}")
            return ""

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # decoded_prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            # print(f"🔧 MEMUPDATE DEBUG: Generated prompt: {decoded_prompt}")
        response_mask, response_logprobs = [], []
        tools_kwargs = kwargs.get("tools_kwargs", {})
        extra_info = kwargs.get("extra_info", {})
        raw_namespace = extra_info.get("namespace", "")
        sample_id = extra_info.get("sample_id", "")
        target_answer = extra_info.get("target_answer", "")
        trial_namespace = raw_namespace + "-" + request_id[:8]

        # 🔧 MEMUPDATE: Create trial-specific namespace and pre-initialize memory store
        if tools_kwargs:
            for tool_name, tool_config in tools_kwargs.items():
                tool_config["create_kwargs"]["trial_namespace"] = trial_namespace
                tool_config["execute_kwargs"]["trial_namespace"] = trial_namespace
        if extra_info:
            extra_info["trial_namespace"] = trial_namespace

        # 🔧 MEMUPDATE: Pre-initialize the store with the trial-specific namespace
        if trial_namespace and sample_id:
            try:
                from memupdate.tools.base_memory_tool import MemoryStoreManager

                MemoryStoreManager.init_conversation_memory(trial_namespace, sample_id)
            except Exception as e:
                print(f"⚠️ [AgentLoop] Failed to pre-initialize memory store: {e}")
        
        # 🔧 MEMUPDATE: Retrieve initial RAG context and add to messages
        target_question = extra_info.get("target_question", "")
        if trial_namespace and sample_id and target_question:
            initial_context = await self._retrieve_initial_context(target_question, sample_id, trial_namespace)
            if initial_context:
                # Add context as a user message
                context_message = {
                    "role": "user", 
                    "content": f"Here is some relevant context from the conversation database that may help answer the question:\n\n{initial_context}\n\nNow, please search for more specific information and submit your final answer using the submit_answer tool."
                }
                messages.append(context_message)
                
                # Regenerate prompt_ids with the updated messages
                if self.processor is not None:
                    raw_prompt = await self.loop.run_in_executor(
                        None,
                        lambda: self.processor.apply_chat_template(
                            messages,
                            tools=self.tool_schemas,
                            add_generation_prompt=True,
                            tokenize=False,
                            **self.apply_chat_template_kwargs,
                        ),
                    )
                    model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
                    prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
                else:
                    prompt_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.apply_chat_template(
                            messages,
                            tools=self.tool_schemas,
                            add_generation_prompt=True,
                            tokenize=True,
                            **self.apply_chat_template_kwargs,
                        ),
                    )

        user_turns, assistant_turns = 0, 0
        termination_reason = "COMPLETED"  # Default to completed
        answer_submitted = False  # Track if submit_answer was called
        while True:
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
                )
            response_ids = output.token_ids
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            if output.log_probs:
                response_logprobs += output.log_probs
            assistant_turns += 1

            # reach max response length
            if len(response_mask) >= self.response_length:
                decoded_prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                termination_reason = "MAX_RESPONSE_LENGTH"
                break

            # reach max assistant turns
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                decoded_prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                termination_reason = "MAX_ASSISTANT_TURNS"
                break

            # reach max user turns
            if self.max_user_turns and user_turns >= self.max_user_turns:
                decoded_prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                termination_reason = "MAX_USER_TURNS"
                break

            # no tool calls
            content, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            if not tool_calls:
                decoded_prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                termination_reason = "COMPLETED"  # Natural completion
                break

            # call tools
            tasks = []
            actual_tool_calls = tool_calls[: self.max_parallel_calls]
            for tool_call in actual_tool_calls:
                tasks.append(self._call_tool(tool_call, tools_kwargs))
            with simple_timer("tool_calls", metrics):
                tool_responses = await asyncio.gather(*tasks)
            
            # Check if submit_answer was called
            for tool_call in actual_tool_calls:
                if hasattr(tool_call, 'name') and tool_call.name == 'submit_answer':
                    answer_submitted = True
                    termination_reason = "ANSWER_SUBMITTED"
            
            if any(isinstance(item, Exception) for item in tool_responses):
                break

            # Extract messages and update multi_modal_data
            tool_messages = []
            new_images_this_turn = []
            
            # Calculate remaining turns info
            remaining_assistant = f"turns remaining: {self.max_assistant_turns - assistant_turns}" if self.max_assistant_turns else ""
            
            status_info = f"\n[{remaining_assistant}]"
            
            for tool_response in tool_responses:
                # Create message from tool response
                if tool_response.image or tool_response.video:
                    # Multi-modal content with structured format
                    content = []
                    if tool_response.image:
                        content.append({"type": "image"})
                    if tool_response.video:
                        content.append({"type": "video"})
                    if tool_response.text:
                        content.append({"type": "text", "text": (tool_response.text or "") + status_info})
                    else:
                        content.append({"type": "text", "text": status_info})
                    message = {"role": "tool", "content": content}
                else:
                    # Text-only content
                    message = {"role": "tool", "content": (tool_response.text or "") + status_info}

                tool_messages.append(message)

                # Handle image data
                if tool_response.image:
                    if image_data is None:
                        image_data = []
                    elif not isinstance(image_data, list):
                        image_data = [image_data]

                    # Add new image data
                    if isinstance(tool_response.image, list):
                        image_data.extend(tool_response.image)
                        new_images_this_turn.extend(tool_response.image)
                    else:
                        image_data.append(tool_response.image)
                        new_images_this_turn.append(tool_response.image)

                # Handle video data
                if tool_response.video:
                    # Currently not supported, raise informative error
                    logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                    raise NotImplementedError(
                        "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                    )

            # append tool_response_ids
            if self.processor is not None:
                raw_tool_response = await self.loop.run_in_executor(
                    None,
                    lambda messages=tool_messages: self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    ),
                )
                # Use only the new images from this turn for processing tool responses
                current_images = new_images_this_turn if new_images_this_turn else None
                model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
                tool_response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
            else:
                tool_response_ids = await self.loop.run_in_executor(
                    None,
                    lambda messages=tool_messages: self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                    ),
                )
            tool_response_ids = tool_response_ids[len(self.system_prompt) :]

            # NOTE: last turn should not be user turn, or the EOS token reward
            # can't be propagated to previous token in GAE.
            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                termination_reason = "MAX_RESPONSE_LENGTH"
                break

            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(tool_response_ids)
            user_turns += 1
            
            # Break if answer was submitted
            if answer_submitted:
                break

        # Apply overlong_filter if enabled and trajectory was truncated
        if self.overlong_filter and termination_reason in ["MAX_RESPONSE_LENGTH", "MAX_ASSISTANT_TURNS", "MAX_USER_TURNS"]:
            # Mask out the entire response for overlong trajectories
            response_mask = [0] * len(response_mask)
            masked_overlong = True
        else:
            masked_overlong = False

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        multi_modal_data = {"image": image_data} if image_data is not None else {}

        # 🔧 MEMUPDATE: Store trial_namespace and termination info in extra_fields
        extra_fields = {
            "termination_reason": termination_reason,
            "masked_overlong": masked_overlong,
        }
        if trial_namespace:
            extra_fields["trial_namespace"] = trial_namespace

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
            extra_fields=extra_fields,
        )
        return output

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]) -> ToolResponse:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            
            # 🔧 MEMUPDATE: Pass execute_kwargs instead of extra_info for trial_namespace
            tool_execution_response, _, _ = await tool.execute(instance_id, tool_args, **kwargs.get("execute_kwargs", {}))
        except Exception as e:
            if tool_name.lower() == "done":
                return ToolResponse(text="Error when executing tool: 'DONE', please return the string 'DONE' without any other text or formatting.")
            logger.warning(f"Error when executing tool: {e}")
            return ToolResponse(
                text=f"Error when executing tool: {e}",
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs)
