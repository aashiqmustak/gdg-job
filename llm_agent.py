# llm_agent.py

import os
import logging
import json
import re
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.client import configure
from dotenv import load_dotenv
import vertex_search_rag

load_dotenv()

logger = logging.getLogger(__name__)

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

configure(api_key=GEMINI_API_KEY)

def load_company_knowledge():
    """Load company-specific knowledge from files."""
    knowledge = {}
    try:
        # Load tech stack
        with open("gdg-cloud/tech_stack.txt", "r") as f:
            knowledge["tech_stack"] = f.read()
        
        # Load role levels
        with open("gdg-cloud/role_levels.txt", "r") as f:
            knowledge["role_levels"] = f.read()
            
        # Load job description template
        with open("gdg-cloud/job_description_backend.txt", "r") as f:
            knowledge["job_template"] = f.read()
            
    except FileNotFoundError as e:
        logger.warning(f"Company knowledge file not found: {e}")
        knowledge = {
            "tech_stack": "Python, Django, PostgreSQL, React, TypeScript, AWS",
            "role_levels": "Junior (0-2 years), Mid-level (2-5 years), Senior (5+ years)",
            "job_template": "Standard job description template"
        }
    
    return knowledge

def validate_entities(entities):
    """Validate and clean extracted entities."""
    validated = {}
    
    # Job title validation
    if entities.get("job_title"):
        title = entities["job_title"].strip().lower()
        # Common job titles in tech
        valid_titles = [
            "developer", "engineer", "programmer", "architect", "lead", "manager",
            "frontend", "backend", "fullstack", "full stack", "ui", "ux", "designer",
            "devops", "data", "analyst", "scientist", "qa", "tester"
        ]
        if any(valid in title for valid in valid_titles):
            validated["job_title"] = entities["job_title"].strip()
        else:
            validated["job_title"] = None
    else:
        validated["job_title"] = None
    
    # Experience validation
    if entities.get("experience"):
        exp = entities["experience"].strip().lower()
        # Look for years, levels, or seniority indicators
        if re.search(r'\d+\s*years?|junior|senior|mid|level|entry|experienced', exp):
            validated["experience"] = entities["experience"].strip()
        else:
            validated["experience"] = None
    else:
        validated["experience"] = None
    
    # Skills validation
    if entities.get("skills"):
        skills = entities["skills"].strip()
        # Basic validation - should not be empty and should contain actual skills
        if len(skills) > 3 and not skills.lower().startswith("i need"):
            validated["skills"] = skills
        else:
            validated["skills"] = None
    else:
        validated["skills"] = None
    
    # Job type validation
    if entities.get("job_type"):
        job_type = entities["job_type"].strip().lower()
        valid_types = ["full-time", "full time", "part-time", "part time", "contract", "freelance", "remote"]
        if any(valid in job_type for valid in valid_types):
            validated["job_type"] = entities["job_type"].strip()
        else:
            validated["job_type"] = None
    else:
        validated["job_type"] = None
    
    # Location validation
    if entities.get("location"):
        location = entities["location"].strip()
        # Basic validation - should not be empty and should not contain other entity info
        if len(location) > 2 and not any(entity in location.lower() for entity in ["skills:", "experience:", "job_type:"]):
            validated["location"] = location
        else:
            validated["location"] = None
    else:
        validated["location"] = None
    
    return validated

def process_with_adk_agent(query: str, context: str = "") -> str:
    """
    Processes the user query and context using the Google AI Gemini 1.5 Flash model.
    Now includes RAG integration and better validation.
    """
    # Simple fallback for basic hiring detection when API is unavailable
    def simple_hiring_detection(text):
        text_lower = text.lower()
        hiring_keywords = ["hiring", "hire", "looking for", "need", "recruiting", "job opening"]
        if any(keyword in text_lower for keyword in hiring_keywords):
            return True
        return False
    
    # Load company knowledge
    company_knowledge = load_company_knowledge()
    
    # Get relevant context from Vertex AI Search
    rag_context = vertex_search_rag.perform_vertex_ai_search_rag(query)
    
    instruction = """You are a precise hiring intent detection agent. Your task is to:

1. **DETECT INTENT**: Determine if the message is a hiring request
2. **EXTRACT ENTITIES**: Only extract the following entities if they are explicitly mentioned:
   - job_title: The specific job role/title
   - experience: Required experience level or years
   - skills: Required technical skills or technologies
   - job_type: Employment type (full-time, part-time, contract, etc.)
   - location: Work location (remote, specific city, etc.)

3. **VALIDATION RULES**:
   - Only extract entities that are CLEARLY stated in the message
   - If an entity is not explicitly mentioned, set it to null
   - Do NOT infer or guess missing information
   - Do NOT hallucinate or add information not present in the input

4. **RESPONSE FORMAT**: Return ONLY valid JSON with this exact structure:
{
    "intent": "hiring_request" or "not_hiring",
    "entities": {
        "job_title": "exact title mentioned" or null,
        "experience": "exact experience mentioned" or null,
        "skills": "exact skills mentioned" or null,
        "job_type": "exact job type mentioned" or null,
        "location": "exact location mentioned" or null
    }
}

**EXAMPLES**:
Input: "I need a Python developer with 3 years experience"
Output: {
    "intent": "hiring_request",
    "entities": {
        "job_title": "Python developer",
        "experience": "3 years",
        "skills": null,
        "job_type": null,
        "location": null
    }
}

Input: "Hello, how are you?"
Output: {
    "intent": "not_hiring",
    "entities": {
        "job_title": null,
        "experience": null,
        "skills": null,
        "job_type": null,
        "location": null
    }
}
"""
    
    # Build comprehensive context
    context_parts = []
    if context:
        context_parts.append(f"Previous Context: {context}")
    if rag_context:
        context_parts.append(f"Company Knowledge: {rag_context}")
    if company_knowledge:
        context_parts.append(f"Tech Stack: {company_knowledge['tech_stack']}")
        context_parts.append(f"Role Levels: {company_knowledge['role_levels']}")
    
    full_context = "\n\n".join(context_parts) if context_parts else "No additional context available."
    
    full_prompt = f"""**INSTRUCTION:**
{instruction}

**COMPANY CONTEXT:**
{full_context}

**USER MESSAGE:**
{query}

**RESPONSE (JSON ONLY):**
"""

    try:
        logger.info("Sending request to Google AI Gemini 1.5 Flash model.")
        model = GenerativeModel("gemini-1.5-flash")
        
        response = model.generate_content(full_prompt)
        response_text = response.text.strip()
        
        # Try to parse JSON response
        try:
            # Clean the response - remove markdown code blocks if present
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]  # Remove ```
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            
            cleaned_response = cleaned_response.strip()
            
            data = json.loads(cleaned_response)
            
            # Validate the structure
            if "intent" not in data or "entities" not in data:
                raise ValueError("Invalid response structure")
            
            # Validate and clean entities
            if data["intent"] == "hiring_request":
                data["entities"] = validate_entities(data["entities"])
            
            logger.info("Successfully processed and validated LLM response.")
            return json.dumps(data, indent=2)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response_text}")
            # Return a safe fallback
            return json.dumps({
                "intent": "not_hiring",
                "entities": {
                    "job_title": None,
                    "experience": None,
                    "skills": None,
                    "job_type": None,
                    "location": None
                }
            })
            
    except Exception as e:
        logger.error(f"An error occurred while calling the Google AI API: {e}", exc_info=True)
        
        # Check if it's a quota exceeded error
        if "quota" in str(e).lower() or "429" in str(e):
            logger.error("API quota exceeded. Please upgrade your plan or wait for quota reset.")
            return json.dumps({
                "intent": "quota_exceeded",
                "entities": {
                    "job_title": None,
                    "experience": None,
                    "skills": None,
                    "job_type": None,
                    "location": None
                }
            })
        
        # Use simple fallback detection
        is_hiring = simple_hiring_detection(query)
        return json.dumps({
            "intent": "hiring_request" if is_hiring else "not_hiring",
            "entities": {
                "job_title": None,
                "experience": None,
                "skills": None,
                "job_type": None,
                "location": None
            }
        })