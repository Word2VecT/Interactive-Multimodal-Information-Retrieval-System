import os
import base64
import json
import time
from openai import OpenAI
from typing import Optional, Dict, Any, List

# It's recommended to set the API key as an environment variable
# export OPENAI_API_KEY='your_api_key'

# Define the structured information we want to extract
# This remains the single source of truth for our schema
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "model_name": {"type": "string", "description": "The name of the main model or architecture discussed, e.g., 'Llama 3', 'BERT', 'ResNet-50'. If not mentioned, use 'N/A'."},
        "primary_task": {"type": "string", "description": "The primary AI task the paper addresses, e.g., 'Text Generation', 'Image Classification', 'Machine Translation'."},
        "key_contribution": {"type": "string", "description": "A concise, one-sentence summary of the paper's main novelty or contribution."},
        "datasets_used": {"type": "array", "items": {"type": "string"}, "description": "A list of key datasets used for training or evaluation, e.g., ['ImageNet', 'C4', 'WMT14']. If none are mentioned, use an empty list."},
        "evaluation_metrics": {"type": "array", "items": {"type": "string"}, "description": "A list of main evaluation metrics used, e.g., ['BLEU', 'Accuracy', 'Perplexity']. If none are mentioned, use an empty list."},
        "one_sentence_summary": {"type": "string", "description": "A single, overarching summary sentence of the paper's findings and implications."}
    },
    "required": ["model_name", "primary_task", "key_contribution", "datasets_used", "evaluation_metrics", "one_sentence_summary"]
}

def image_to_base64(image_path: str) -> Optional[str]:
    """Converts an image file to a Base64 encoded string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

def is_valid_schema(data: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validates if the dictionary contains all required keys."""
    return all(key in data for key in required_keys)

def extract_information(
    text: str, 
    image_path: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = "gpt-4o-latest",
    retry_times: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Extracts structured information using an LLM, with schema validation and retries.
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        return {"error": f"Client initialization failed: {e}"}

    if not client.api_key:
        return {"error": "API key not set"}

    # --- Reinforced System Prompt ---
    # Explicitly list the required keys in the prompt for redundancy.
    required_keys_str = ", ".join(f'"{key}"' for key in JSON_SCHEMA['required'])
    system_prompt = (
        "You are an expert AI research assistant. Your task is to analyze the provided text (a research paper abstract) "
        "and an optional image. Extract the required information and return it STRICTLY in the specified JSON format. "
        f"The JSON object MUST contain the following keys: {required_keys_str}. "
        "Do not include any explanatory text outside of the JSON object."
    )
    
    user_content = [{"type": "text", "text": text}]
    if image_path:
        base64_image = image_to_base64(image_path)
        if base64_image:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

    for attempt in range(retry_times):
        try:
            print(f"Calling API (model: {model_name}), attempt {attempt + 1}/{retry_times}...")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object", "schema": JSON_SCHEMA},
                temperature=0.1,
            )
            
            extracted_data = json.loads(response.choices[0].message.content)

            # --- Validation Step ---
            if is_valid_schema(extracted_data, JSON_SCHEMA['required']):
                print("API call and schema validation successful.")
                return extracted_data
            else:
                # This will be caught by the exception block's 'else' part
                raise ValueError(f"Returned JSON is missing required keys. Content: {extracted_data}")

        except Exception as e:
            print(f"An error occurred (attempt {attempt + 1}): {e}")
            if attempt < retry_times - 1:
                print("Retrying in 3 seconds...")
                time.sleep(3)
            else:
                print("Max retries reached. API call failed.")
                return {"error": f"API call failed after {retry_times} attempts: {e}"}
    
    return {"error": "An unknown error occurred after all retries."}

if __name__ == '__main__':
    # Example usage for testing the extractor directly
    sample_text = (
        "We introduce Llama 3, a new family of large language models. "
        "Our work focuses on improving pre-training data and scaling laws. "
        "We trained on a new 15T token dataset and evaluated on the MMLU benchmark, "
        "achieving a score of 82.0. This represents a significant improvement in model performance "
        "for few-shot learning scenarios."
    )
    
    # Test with text only
    extracted_info_text = extract_information(text=sample_text, api_key="", base_url="", model_name="chatgpt-4o-latest", retry_times=3)
    print("\n--- Extraction from Text Only ---")
    print(json.dumps(extracted_info_text, indent=2))

    # To test with an image, create a dummy image file named 'dummy.jpg'
    # with open("dummy.jpg", "wb") as f: f.write(b'dummy image data')
    # if os.path.exists("dummy.jpg"):
    #     extracted_info_multimodal = extract_information(text=sample_text, image_path="dummy.jpg")
    #     print("\n--- Extraction from Text and Image ---")
    #     print(json.dumps(extracted_info_multimodal, indent=2))

    # You can now test like this:
    # extract_information(
    #     text="...",
    #     api_key="your_key",
    #     base_url="your_base_url",
    #     model_name="your_model",
    #     retry_times=2
    # ) 