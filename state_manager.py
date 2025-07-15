# state_manager.py
import os
import json
import logging

logger = logging.getLogger(__name__)
SAVED_JOBS_FILE = "saved_jobs.json"

def save_job_to_file(entities: dict):
    """Saves a completed job entity dictionary to the JSON file."""
    try:
        # Read existing jobs
        if os.path.exists(SAVED_JOBS_FILE):
            with open(SAVED_JOBS_FILE, 'r') as f:
                jobs = json.load(f)
        else:
            jobs = []
        # Add the new job
        jobs.append(entities)
        # Write back to the file
        with open(SAVED_JOBS_FILE, 'w') as f:
            json.dump(jobs, f, indent=4)
        logger.info(f"Successfully saved job to {SAVED_JOBS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save job to file: {e}", exc_info=True)
        return False