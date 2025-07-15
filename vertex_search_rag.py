# vertex_search_rag.py

import os
import logging
from google.cloud import discoveryengine_v1beta as discoveryengine
from google.api_core.client_options import ClientOptions
from google.auth.exceptions import DefaultCredentialsError

logger = logging.getLogger(__name__)

def perform_vertex_ai_search_rag(query_text: str) -> str:
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
    data_store_id = os.environ.get("GOOGLE_CLOUD_DATA_STORE_ID")

    # --- THIS IS THE KEY CHANGE ---
    # Your Data Store is in the 'global' region. We must specify this.
    # The API endpoint for global data stores is 'discoveryengine.googleapis.com'
    location = "global"
    api_endpoint = f"{location}-discoveryengine.googleapis.com"

    if not project_id or not data_store_id:
        logger.error("Vertex AI Search environment variables not fully set.")
        return ""

    try:
        # Set the correct API endpoint for the 'global' location
        client_options = ClientOptions(api_endpoint=api_endpoint)
        client = discoveryengine.SearchServiceClient(client_options=client_options)

        serving_config = client.serving_config_path(
            project=project_id,
            location=location,
            data_store=data_store_id,
            serving_config="default_config",
        )

        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query_text,
            page_size=3,
        )

        response = client.search(request)

        # ... (rest of the file is the same)
        retrieved_content_parts = []
        for result in response.results:
            doc_data = result.document.derived_struct_data
            content = doc_data.get("content") or ""
            title = doc_data.get("title") or ""
            url = doc_data.get("url") or ""

            if content:
                formatted_part = f"--- Retrieved Document ---\nTitle: {title}\nURL: {url}\nContent: {content}\n"
                retrieved_content_parts.append(formatted_part)

        if retrieved_content_parts:
            logger.info(f"Successfully retrieved {len(retrieved_content_parts)} relevant documents.")
            return "\n\n".join(retrieved_content_parts)
        else:
            logger.info(f"No relevant content found in Vertex AI Search for query.")
            return ""

    except DefaultCredentialsError:
        logger.error(
            "Google Cloud Authentication FAILED. "
            "Please run 'gcloud auth application-default login' in your terminal."
        )   
        return ""
    except Exception as e:
        logger.error(f"An error occurred during Vertex AI Search RAG: {e}", exc_info=True)
        return ""
    
    