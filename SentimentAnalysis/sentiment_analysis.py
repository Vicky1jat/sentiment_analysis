"""
This module provides a function to analyze sentiment using the IBM Watson NLP API.
"""

import requests  # Third-party import
# Removed `import json` as it was unused

def sentiment_analyzer(text_to_analyse):
    """
    Analyze sentiment of the provided text using Watson NLP sentiment API.

    Args:
        text_to_analyse (str): The input text to analyze.

    Returns:
        dict: A dictionary with 'label' and 'score' if successful, else 'error'.
    """
    url = (
        "https://sn-watson-sentiment-bert.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/SentimentPredict"
    )

    payload = {
        "raw_document": {
            "text": text_to_analyse
        }
    }

    headers = {
        "grpc-metadata-mm-model-id": "sentiment_aggregated-bert-workflow_lang_multi_stock"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        result = response.json()  # Use requests' built-in JSON parsing

        label = result.get("documentSentiment", {}).get("label")
        score = result.get("documentSentiment", {}).get("score")

        return {"label": label, "score": score}

    except requests.exceptions.RequestException as error:
        return {"error": str(error)}
