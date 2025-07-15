# job_description_agent.py

import os
import logging
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.client import configure
import vertex_search_rag

logger = logging.getLogger(__name__)

# This agent uses the same API key configured in your main app
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Configure the API key
configure(api_key=GEMINI_API_KEY)

def load_company_knowledge():
    """Load company-specific knowledge for job descriptions."""
    knowledge = {}
    try:
        with open("gdg-cloud/tech_stack.txt", "r") as f:
            knowledge["tech_stack"] = f.read()
        with open("gdg-cloud/role_levels.txt", "r") as f:
            knowledge["role_levels"] = f.read()
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

def generate_job_description(job_title: str, experience: str, skills: str, job_type: str, location: str) -> str:
    """
    Generates a job description using the Gemini model with company knowledge.
    """
    # Load company knowledge
    company_knowledge = load_company_knowledge()
    
    # Get relevant context from Vertex AI Search
    search_query = f"{job_title} {skills} {experience}"
    rag_context = vertex_search_rag.perform_vertex_ai_search_rag(search_query)
    
    # Build comprehensive prompt
    prompt = f"""You are a professional job description writer. Create a LinkedIn job post based on the provided information.

**COMPANY CONTEXT:**
Tech Stack: {company_knowledge['tech_stack']}
Role Levels: {company_knowledge['role_levels']}
Job Template: {company_knowledge['job_template']}

**RAG CONTEXT:**
{rag_context if rag_context else "No additional context available."}

**JOB REQUIREMENTS:**
- Title: {job_title}
- Experience: {experience}
- Skills: {skills}
- Job Type: {job_type}
- Location: {location}

**INSTRUCTIONS:**
1. Use ONLY the information provided above
2. Do NOT add fictional company details, benefits, or requirements not mentioned
3. Keep the tone professional and engaging
4. Ensure the description is under 2500 characters
5. Include relevant hashtags for LinkedIn
6. Structure the post with clear sections

**OUTPUT FORMAT:**
Create a LinkedIn post with:
- Engaging headline
- Brief company intro (generic, don't make up company details)
- Role description based on provided requirements
- Required skills and experience
- Job type and location
- Call to action
- Relevant hashtags

**IMPORTANT:** Only use the information provided. Do not hallucinate company names, specific benefits, or requirements not mentioned in the input.
"""

    try:
        logger.info("Sending request to generate job description.")
        model = GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        # Validate response length
        description = response.text.strip()
        if len(description) > 2500:
            description = description[:2500] + "..."
        
        logger.info("Successfully received job description from model.")
        return description
        
    except Exception as e:
        logger.error(f"An error occurred while generating the job description: {e}", exc_info=True)
        return f"""🚀 We're Hiring!

📌 {job_title}
📍 Location: {location}
⏰ Type: {job_type}
🧠 Experience: {experience}
🛠 Skills: {skills}

We're looking for a talented {job_title} to join our team. If you have the required experience and skills, we'd love to hear from you!

#Hiring #JobOpening #Careers"""