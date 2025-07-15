import os
import logging
import asyncio # New: For asynchronous programming
import time
import json
from collections import deque
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, ValidationError # Pydantic for structured data
from slack_sdk.web import WebClient
from slack_sdk.socket_mode.aiohttp import SocketModeClient as AsyncSocketModeClient # New: Async Slack client
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

# Import the new Vertex AI Search RAG module
import vertex_search_rag

# New: Imports for Google ADK
import adk
from adk.behaviours import Behavior
from adk.prompts import PromptProvider
from adk.llms import LLM

load_dotenv() # Load environment variables from .env file

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Batch Processing Configuration ---
BATCH_SIZE = 5
BATCH_TIMEOUT_SECONDS = 10

# Async-safe queue for messages
message_queue = deque()
# No explicit lock needed for deque with single-consumer asyncio, but good practice if producers are multiple tasks.
# For simplicity here, we'll rely on the nature of `asyncio.sleep` yielding control.
# If extreme concurrency is expected for `message_queue` writes, an `asyncio.Lock` would be used.

# Async timer task
batch_task = None
last_message_time = time.time() # To track when the last message arrived

# --- Slack Initialization ---
slack_bot_token = os.environ["SLACK_BOT_TOKEN"]
slack_client = WebClient(token=slack_bot_token) # WebClient can remain sync for direct API calls

# Note: SocketModeClient is now AsyncSocketModeClient from aiohttp package
socket_mode_client = AsyncSocketModeClient(
    app_token=os.environ["SLACK_APP_TOKEN"],
    web_client=slack_client
)

# --- Pydantic Models for Structured Output (Same as before) ---
class HiringRequestEntities(BaseModel):
    job_title: Optional[str] = Field(None, description="The title of the job being hired for, e.g., 'front end developer', 'UI/UX designer'. Null if not specified or not a hiring request.")
    job_type: Optional[str] = Field(None, description="The type of job, e.g., 'full-time', 'part-time', 'remote', 'contract'. Null if not specified.")
    skills: Optional[str] = Field(None, description="Key skills required for the job, e.g., 'Figma', 'Python', 'React'. Can be a comma-separated string if multiple. Null if not specified.")
    experience: Optional[str] = Field(None, description="Required years of experience, e.g., '4 years', 'junior', 'senior'. Null if not specified.")

class LlmAgentOutput(BaseModel):
    intent: str = Field(description="The detected intent of the message. 'hiring_request' if it's a job posting, otherwise 'other'.")
    entities: HiringRequestEntities = Field(description="Extracted entities if the intent is 'hiring_request'. All fields will be null if intent is 'other'.")

# --- ADK Agent Behavior Definition ---
class IntentEntityBehavior(Behavior):
    def __init__(self, output_schema_str: str, instruction: str):
        super().__init__(
            name="IntentEntityBehavior",
            description="Detects hiring requests and extracts job-related entities in JSON format.",
        )
        self.output_schema_str = output_schema_str
        self.base_instruction = instruction

    async def generate_response(self, prompt_provider: PromptProvider) -> str:
        """
        Defines how this behavior generates a response using the LLM.
        """
        user_message = prompt_provider.context.get("user_message", "")
        retrieved_context = prompt_provider.context.get("retrieved_context", "No additional context found.")

        # Construct the detailed prompt, including the base instruction and JSON schema
        system_instruction = f"""{self.base_instruction}

You must strictly adhere to the following JSON output format. Ensure the output is *only* the JSON object, without any surrounding text or markdown backticks.
```json
{self.output_schema_str}