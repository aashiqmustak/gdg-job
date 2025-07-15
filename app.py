import os
import logging
import threading
from collections import deque
import time
from threading import Event
import json
from typing import Any

from dotenv import load_dotenv
from slack_sdk.web import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

load_dotenv()

import llm_agent
import job_description_agent
import state_manager
import linkedin_poster

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Slack and State Configuration ---
slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
bot_user_id = slack_client.auth_test()["user_id"]
socket_mode_client = SocketModeClient(app_token=os.environ["SLACK_APP_TOKEN"], web_client=slack_client)

# The new state will store a list of messages for the conversation history
conversation_states = {}
# Track recently completed conversations to prevent immediate restarts
recently_completed = {} 

def validate_llm_response(llm_response: str) -> dict:
    """Validate and parse LLM response safely."""
    try:
        # Try to parse as JSON
        data = json.loads(llm_response)
        
        # Validate structure
        if not isinstance(data, dict):
            raise ValueError("Response is not a dictionary")
        
        if "intent" not in data:
            raise ValueError("Missing 'intent' field")
        
        if "entities" not in data:
            raise ValueError("Missing 'entities' field")
        
        # Ensure entities is a dictionary
        if not isinstance(data["entities"], dict):
            raise ValueError("Entities field is not a dictionary")
        
        # Validate entity fields
        required_entities = ["job_title", "experience", "skills", "job_type", "location"]
        for entity in required_entities:
            if entity not in data["entities"]:
                data["entities"][entity] = None
        
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Raw response: {llm_response}")
        return {
            "intent": "not_hiring",
            "entities": {
                "job_title": None,
                "experience": None,
                "skills": None,
                "job_type": None,
                "location": None
            }
        }
    except Exception as e:
        logger.error(f"Error validating LLM response: {e}")
        return {
            "intent": "not_hiring",
            "entities": {
                "job_title": None,
                "experience": None,
                "skills": None,
                "job_type": None,
                "location": None
            }
        }

def process_conversation(channel_id: str, user_id: str):
    """
    Takes the current conversation history, re-evaluates it with the LLM,
    and decides on the next step (ask, confirm, or finalize).
    """
    if channel_id not in conversation_states or not conversation_states[channel_id]["messages"]:
        return

    state = conversation_states[channel_id]
    messages = state["messages"]
    
    # Combine the message history into a single query for the LLM
    full_conversation_text = "\n".join(messages)
    logger.info(f"Re-evaluating conversation for channel {channel_id}:\n{full_conversation_text}")

    # The LLM gets the full conversation to re-evaluate entities
    llm_response = llm_agent.process_with_adk_agent(query=full_conversation_text, context="")
    logger.info(f"LLM Agent Response:\n{llm_response}")

    # Validate the LLM response
    data = validate_llm_response(llm_response)
    
    # Handle quota exceeded error
    if data.get("intent") == "quota_exceeded":
        slack_client.chat_postMessage(channel=channel_id, text="⚠️ I've reached my daily API limit. Please try again tomorrow or contact support to upgrade the plan.")
        return
    
    if data.get("intent") == "hiring_request" and "entities" in data:
        entities = data["entities"]
        
        # Log the extracted entities for debugging
        logger.info(f"Extracted entities: {json.dumps(entities, indent=2)}")
        
        # --- This logic is now centralized and more robust ---
        follow_up_order = ["job_title", "experience", "skills", "job_type", "location"]
        first_missing_entity = next((entity for entity in follow_up_order if entities.get(entity) is None), None)

        if first_missing_entity:
            # If an entity is still missing, ask for it
            questions = {
                "job_title": "What is the job title/role you're hiring for?",
                "experience": "What is the required level of experience (e.g., 2 years, Senior)?",
                "skills": "What specific skills are required for this role?",
                "job_type": "Is this a full-time, part-time, or contract position?",
                "location": "What is the location for this role (e.g., Remote, specific city)?"
            }
            question_to_ask = questions[first_missing_entity]
            slack_client.chat_postMessage(channel=channel_id, text=question_to_ask)
            logger.info(f"Asked follow-up for '{first_missing_entity}'.")
        else:
            # All entities are filled, finalize the process
            logger.info("All entities are filled. Finalizing conversation.")
            
            # Validate that we have meaningful data
            if not entities.get("job_title") or entities.get("job_title").strip() == "":
                slack_client.chat_postMessage(channel=channel_id, text="I couldn't determine the job title. Could you please specify what role you're hiring for?")
                return
            
            # Check if we're in edit mode and show original vs new details
            if state.get("editing", False):
                original_entities = state.get("original_entities", {})
                confirmation_text = (
                    "✏️ Here are the updated job details:\n"
                    f"  • *Job Title*: {entities.get('job_title', 'N/A')} (was: {original_entities.get('job_title', 'N/A')})\n"
                    f"  • *Experience*: {entities.get('experience', 'N/A')} (was: {original_entities.get('experience', 'N/A')})\n"
                    f"  • *Skills*: {entities.get('skills', 'N/A')} (was: {original_entities.get('skills', 'N/A')})\n"
                    f"  • *Job Type*: {entities.get('job_type', 'N/A')} (was: {original_entities.get('job_type', 'N/A')})\n"
                    f"  • *Location*: {entities.get('location', 'N/A')} (was: {original_entities.get('location', 'N/A')})"
                )
            else:
                confirmation_text = (
                    "Great! I have all the details. Here's the complete hiring request:\n"
                    f"  • *Job Title*: {entities.get('job_title', 'N/A')}\n"
                    f"  • *Experience*: {entities.get('experience', 'N/A')}\n"
                    f"  • *Skills*: {entities.get('skills', 'N/A')}\n"
                    f"  • *Job Type*: {entities.get('job_type', 'N/A')}\n"
                    f"  • *Location*: {entities.get('location', 'N/A')}"
                )
            slack_client.chat_postMessage(channel=channel_id, text=confirmation_text)
            
            slack_client.chat_postMessage(channel=channel_id, text="Now, I will generate a job description...")
            
            try:
                job_description = job_description_agent.generate_job_description(**entities)
                cleaned_description = job_description.replace('**', '')

                message_blocks = [
                    {"type": "section", "text": {"type": "mrkdwn", "text": cleaned_description}},
                    {"type": "actions", "block_id": "jd_approval_buttons", "elements": [
                        {"type": "button", "text": {"type": "plain_text", "text": "Post Job"}, "style": "primary", "action_id": "jd_post_yes"},
                        {"type": "button", "text": {"type": "plain_text", "text": "Draft"}, "action_id": "jd_draft"},
                        {"type": "button", "text": {"type": "plain_text", "text": "Edit"}, "action_id": "jd_edit"},
                        {"type": "button", "text": {"type": "plain_text", "text": "No"}, "style": "danger", "action_id": "jd_no"}
                    ]}
                ]
                slack_client.chat_postMessage(channel=channel_id, blocks=message_blocks, text="Job description generated.")
                
                # Store the final entities so the button handler can use them
                conversation_states[channel_id]["final_entities"] = entities
                
            except Exception as e:
                logger.error(f"Error generating job description: {e}", exc_info=True)
                slack_client.chat_postMessage(channel=channel_id, text="Sorry, I encountered an error while generating the job description. Please try again.")
    else:
        # Not a hiring request or invalid response
        if data.get("intent") == "not_hiring":
            slack_client.chat_postMessage(channel=channel_id, text="I didn't detect a hiring request. If you're looking to hire someone, please let me know the job details!")
        else:
            slack_client.chat_postMessage(channel=channel_id, text="I'm having trouble understanding your request. Could you please clarify if you're looking to hire someone?")


def process_slack_events(client: Any, req: SocketModeRequest):
    if req.type == "events_api" and req.payload.get("event", {}).get("type") == "message":
        event = req.payload.get("event", {})
        user_id = event.get("user")
        if user_id == bot_user_id or "subtype" in event:
            return

        try:
            channel_id, message_text = event.get("channel"), event.get("text", "")
            logger.info(f"Received message from user {user_id} in channel {channel_id}: '{message_text}'")

            # If there's an active conversation in the channel, add the new message to its history
            if channel_id in conversation_states:
                # Ensure the reply is from the same user who started the conversation
                if user_id == conversation_states[channel_id].get("user_id"):
                    # Check if this is a duplicate message
                    if message_text not in conversation_states[channel_id]["messages"]:
                        conversation_states[channel_id]["messages"].append(message_text)
                        logger.info("Appended message to existing conversation history.")
                        # Process the conversation with the updated history
                        process_conversation(channel_id, user_id)
                    else:
                        logger.info("Duplicate message detected, skipping processing.")
                else:
                    # If a different user chimes in, start a new conversation for them
                    conversation_states[channel_id] = {"user_id": user_id, "messages": [message_text]}
                    logger.info("New user in channel, starting a new conversation state.")
                    # Process the conversation with the updated history
                    process_conversation(channel_id, user_id)
            else:
                # If no active conversation, start a new one
                conversation_states[channel_id] = {"user_id": user_id, "messages": [message_text]}
                logger.info("No existing state found. Started a new conversation state.")
                # Process the conversation with the updated history
                process_conversation(channel_id, user_id)

        except Exception as e:
            logger.error(f"Error in message processing: {e}", exc_info=True)


def handle_interaction(client: Any, req: SocketModeRequest):
    if req.type == "interactive":
        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
        payload = req.payload
        action_id, channel_id, user_id = payload["actions"][0]["action_id"], payload["channel"]["id"], payload["user"]["id"]

        if channel_id in conversation_states and user_id == conversation_states[channel_id].get("user_id"):
            entities = conversation_states[channel_id].get("final_entities")
            if not entities:
                slack_client.chat_postMessage(channel=channel_id, text="Sorry, I seem to have lost the context. Please start over.")
                return

            if action_id == "jd_post_yes":
                try:
                    success, message = linkedin_poster.post_job(entities)
                    if success:
                        response_text = f"✅ Job posted successfully! Link: {message}"
                        slack_client.chat_postMessage(channel=channel_id, text=response_text)
                        slack_client.chat_postMessage(channel=channel_id, text="🎉 Job posted! Conversation ended.")
                        # Mark as recently completed to prevent any further processing
                        recently_completed[channel_id] = True
                        del conversation_states[channel_id]
                        return
                    else:
                        response_text = f"❌ Error posting to LinkedIn: {message}"
                        slack_client.chat_postMessage(channel=channel_id, text=response_text)
                except Exception as e:
                    logger.error(f"Error posting to LinkedIn: {e}", exc_info=True)
                    slack_client.chat_postMessage(channel=channel_id, text="❌ An error occurred while posting to LinkedIn. Please try again.")
            
            elif action_id == "jd_draft":
                try:
                    job_id = state_manager.save_job_to_file(entities)
                    if job_id:
                        slack_client.chat_postMessage(channel=channel_id, text=f"✅ Job saved as draft. Job ID: {job_id}")
                        slack_client.chat_postMessage(channel=channel_id, text="💾 Draft saved! Conversation ended.")
                        # Mark as recently completed to prevent any further processing
                        recently_completed[channel_id] = True
                        del conversation_states[channel_id]
                        return
                    else:
                        slack_client.chat_postMessage(channel=channel_id, text="❌ An error occurred while saving the draft. Please try again.")
                except Exception as e:
                    logger.error(f"Error saving draft: {e}", exc_info=True)
                    slack_client.chat_postMessage(channel=channel_id, text="❌ An error occurred while saving the draft. Please try again.")
            
            elif action_id == "jd_edit":
                try:
                    # Reset conversation to allow editing
                    slack_client.chat_postMessage(channel=channel_id, text="✏️ Let's edit the job details. Please provide the updated information:")
                    
                    # Clear the processed flag to allow new messages
                    if channel_id in conversation_states:
                        conversation_states[channel_id]["processed"] = False
                        # Keep the entities for reference but allow new input
                        conversation_states[channel_id]["editing"] = True
                        conversation_states[channel_id]["original_entities"] = entities.copy()
                    
                    return
                except Exception as e:
                    logger.error(f"Error starting edit mode: {e}", exc_info=True)
                    slack_client.chat_postMessage(channel=channel_id, text="❌ An error occurred while starting edit mode. Please try again.")
            
            elif action_id == "jd_no":
                try:
                    # Cancel the job posting and reset conversation
                    slack_client.chat_postMessage(channel=channel_id, text="❌ Job posting cancelled.")
                    slack_client.chat_postMessage(channel=channel_id, text="🔄 Starting fresh. Please provide new job details:")
                    
                    # Clear conversation state to start over
                    del conversation_states[channel_id]
                    return
                except Exception as e:
                    logger.error(f"Error cancelling job posting: {e}", exc_info=True)
                    slack_client.chat_postMessage(channel=channel_id, text="❌ An error occurred while cancelling. Please try again.")
            
            # Only reach here if there was an error
            del conversation_states[channel_id]
        else:
            logger.warning("Received button click but no active conversation found.")

if __name__ == "__main__":
    try:
        socket_mode_client.socket_mode_request_listeners.append(process_slack_events)
        socket_mode_client.socket_mode_request_listeners.append(handle_interaction)
        
        print("🤖 Slack Bot is starting...")
        socket_mode_client.connect()
        Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutdown initiated...")
    finally:
        logger.info("Application terminated.")