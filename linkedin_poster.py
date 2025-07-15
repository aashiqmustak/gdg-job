# linkedin_poster.py

import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

API_URL = "https://api.linkedin.com/v2/ugcPosts"

def post_job(job_details: dict) -> tuple[bool, str]:
    """
    Posts a job to LinkedIn.
    Returns a tuple of (success_status, message_or_url).
    """
    access_token = os.environ.get("LINKEDIN_ACCESS_TOKEN")
    author_urn = os.environ.get("PERSON_URN")

    if not all([access_token, author_urn]):
        error_msg = "LinkedIn credentials (LINKEDIN_ACCESS_TOKEN or PERSON_URN) not found."
        logger.error(error_msg)
        return False, error_msg

    # Format the post text from the job details
    post_text = (
        f"🚀 New Job Opportunity!\n\n"
        f"📌 Title: {job_details.get('job_title', 'N/A')}\n"
        f"🧠 Experience: {job_details.get('experience', 'N/A')}\n"
        f"📍 Location: {job_details.get('location', 'N/A')}\n"
        f"🛠 Skills: {job_details.get('skills', 'N/A')}\n"
        f"Job Type: {job_details.get('job_type', 'N/A')}\n\n"
        "#Hiring #JobOpening #Careers"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0"
    }

    payload = {
        "author": author_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": post_text},
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }

    try:
        logger.info("--- SENDING REQUEST TO LINKEDIN API ---")
        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 201:
            post_id = response.headers.get("x-restli-id", "unknown")
            post_url = f"https://www.linkedin.com/feed/update/{post_id}"
            logger.info(f"Successfully posted to LinkedIn: {post_url}")
            return True, post_url
        else:
            error_details = f"Failed: {response.status_code} - {response.text}"
            logger.error(error_details)
            return False, error_details

    except Exception as e:
        logger.error(f"An exception occurred while posting to LinkedIn: {e}", exc_info=True)
        return False, str(e)
    print("hi")