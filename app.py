import gradio as gr
from retriever import Retriever
from database import Database
import shutil
import os
from PIL import Image
from datetime import datetime
import traceback
import json
from functools import partial
import base64
from io import BytesIO
from openai_extractor import extract_information

# --- OpenAI Configuration ---
# Modify these values as needed
OPENAI_API_KEY = ""  # Use environment variable by default
OPENAI_BASE_URL = ""  # e.g., "https://api.example.com/v1"
OPENAI_MODEL_NAME = "chatgpt-4o-latest"
OPENAI_RETRY_TIMES = 10

# --- Feedback Persistence ---
SEARCH_FEEDBACK_FILE = "search_feedback.json"
EXTRACTION_FEEDBACK_FILE = "extraction_feedback.json"


def load_feedback_scores(filename):
    """Loads feedback scores from a specified JSON file."""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                scores = json.load(f)
                return scores.get("total", 0), scores.get("accurate", 0)
            except json.JSONDecodeError:
                return 0, 0
    return 0, 0


def save_feedback_scores(filename, total, accurate):
    """Saves feedback scores to a specified JSON file."""
    with open(filename, "w") as f:
        json.dump({"total": total, "accurate": accurate}, f)


def image_to_base64(image_path):
    """Converts an image file to a Base64 data URI."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        # You might need to adjust the mime type depending on your images
        # Common types: image/jpeg, image/png, image/gif
        mime_type = "image/png"
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""


# --- Initialization ---
if not os.path.exists("data"):
    os.makedirs("data")

db = Database()
retriever = Retriever()

# --- Functions for Gradio Interface ---


def add_item(item_type, title, content, url, image_path):
    """Adds a new item to the database after generating its embedding and extracting info."""
    try:
        print(f"--- Received request to add item of type: {item_type} ---")
        if not title or (item_type in ["text", "image-text"] and not content):
            error_msg = "Title and Content are required for this item type."
            print(f"Validation failed: {error_msg}")
            return error_msg, error_msg, title, content, url, image_path

        embedding = None
        saved_image_path = None
        final_content = content
        text_for_extraction = content
        image_for_extraction = None

        if item_type == "text":
            print(f"Generating text embedding for title: {title}")
            embedding = retriever.get_text_embedding(content)
            print("Text embedding generated successfully.")

        elif item_type == "image":
            if image_path is None:
                error_msg = "Image is required for item type 'image'."
                print(f"Validation failed: {error_msg}")
                return error_msg, error_msg, title, content, url, image_path
            saved_image_path = os.path.join("data", os.path.basename(image_path))
            shutil.copy(image_path, saved_image_path)
            final_content = saved_image_path
            print(f"Generating image embedding for image: {saved_image_path}")
            embedding = retriever.get_image_embedding(saved_image_path)
            print("Image embedding generated successfully.")
            text_for_extraction = title
            image_for_extraction = saved_image_path

        elif item_type == "image-text":
            if image_path is None or not content:
                error_msg = "Image and Content are required for item type 'image-text'."
                print(f"Validation failed: {error_msg}")
                return error_msg, error_msg, title, content, url, image_path
            saved_image_path = os.path.join("data", os.path.basename(image_path))
            shutil.copy(image_path, saved_image_path)
            final_content = f"{content} | {saved_image_path}"
            print(f"Generating image-text embedding for: {final_content}")
            embedding = retriever.get_image_text_embedding(saved_image_path, content)
            print("Image-text embedding generated successfully.")
            text_for_extraction = content
            image_for_extraction = saved_image_path

        if embedding is not None:
            print("--- Starting Information Extraction ---")
            extracted_info = extract_information(
                text=text_for_extraction,
                image_path=image_for_extraction,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                model_name=OPENAI_MODEL_NAME,
                retry_times=OPENAI_RETRY_TIMES,
            )
            extracted_info_json = json.dumps(extracted_info) if extracted_info else "{}"
            print(f"--- Extraction Complete. Result: {extracted_info_json} ---")

            current_date = datetime.now().isoformat()
            print(f"Adding item to database...")
            db.add(item_type, title, final_content, url, current_date, embedding, extracted_info_json)
            print("Item added to database successfully.")
            added_details = (
                f"Type: {item_type}\nTitle: {title}\n"
                f"Content: {final_content}\nURL: {url}\nDate: {current_date}\n"
                f"Extracted Info: {json.dumps(extracted_info, indent=2)}"
            )
            # On success, clear all input fields for the next entry
            return f"'{title}' added and info extracted successfully!", added_details, "", "", "", None

        # This case might happen if embedding generation fails unexpectedly
        print("Failed to add item because embedding was not generated.")
        return "Failed to add item. Check logs.", "", title, content, url, image_path

    except Exception as e:
        print("!!!!!!!! AN ERROR OCCURRED IN add_item !!!!!!!!")
        print(traceback.format_exc())
        error_message = f"An error occurred: {e}"
        # On exception, keep inputs for user to retry
        return error_message, traceback.format_exc(), title, content, url, image_path


def record_search_feedback(choice, total, accurate):
    """Updates the search accuracy score and saves it."""
    total += 1
    if choice == "accurate":
        accurate += 1
    save_feedback_scores(SEARCH_FEEDBACK_FILE, total, accurate)

    if total > 0:
        accuracy_percent = (accurate / total) * 100
        accuracy_text = f"**Search Accuracy:** {accuracy_percent:.1f}% ({accurate} / {total} votes)"
    else:
        accuracy_text = "**Search Accuracy:** N/A"
    return total, accurate, accuracy_text


def record_extraction_feedback(choice, total, accurate):
    """Updates the extraction accuracy score and saves it."""
    total += 1
    if choice == "correct":
        accurate += 1
    save_feedback_scores(EXTRACTION_FEEDBACK_FILE, total, accurate)

    if total > 0:
        accuracy_percent = (accurate / total) * 100
        accuracy_text = f"**Extraction Accuracy:** {accuracy_percent:.1f}% ({accurate} / {total} votes)"
    else:
        accuracy_text = "**Extraction Accuracy:** N/A"
    return total, accurate, accuracy_text


def format_extracted_info_html(info_json_str: str) -> str:
    """Formats the extracted info JSON into a nice HTML block."""
    try:
        info = json.loads(info_json_str)
        # We still check for major errors, but will now display all fields
        if not info or "error" in info:
            # Display the error if one occurred during extraction
            error_msg = info.get("error", "Unknown extraction error")
            return f"<div style='background-color:#f8d7da; border-left: 4px solid #dc3545; padding: 10px; margin-top: 10px; font-size: 0.9em;'><strong>Extraction Failed:</strong> {error_msg}</div>"

        html = "<div style='background-color:#f0f0f0; border-left: 4px solid #007bff; padding: 10px; margin-top: 10px; font-size: 0.9em;'>"
        html += "<strong>Extracted Information:</strong><ul>"

        # Define user-friendly labels for each key
        labels = {
            "model_name": "Model Name",
            "primary_task": "Primary Task",
            "key_contribution": "Key Contribution",
            "datasets_used": "Datasets",
            "evaluation_metrics": "Metrics",
            "one_sentence_summary": "Summary",
        }

        for key, label in labels.items():
            value = info.get(key)

            # Set a default value for display if the original value is empty, None, or an empty list
            display_value = "N/A"
            if value:
                display_value = ", ".join(value) if isinstance(value, list) else value

            html += f"<li><strong>{label}:</strong> {display_value}</li>"

        html += "</ul></div>"
        return html
    except (json.JSONDecodeError, TypeError):
        # This handles cases where the stored string is not valid JSON
        return "<div style='background-color:#fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin-top: 10px; font-size: 0.9em;'><strong>Note:</strong> Could not parse extracted information.</div>"


def search_items(query, image_query_path, top_k):
    """Performs a search and returns formatted results."""
    # Convert Gradio temp path to a usable path
    temp_image_path = image_query_path  # gr.Image with type="filepath" gives a string path

    results = retriever.search(query, temp_image_path, int(top_k))

    if not results:
        # If no results, hide the feedback buttons
        return "<p>No results found.</p>", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    output_html = "<div>"
    for sim, item in results:
        output_html += "<div style='border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 5px;'>"
        output_html += f"<h3>{item.get('title', 'N/A')} (Score: {sim:.4f})</h3>"

        item_content = item.get("content", "")
        item_type = item.get("type", "")

        if item_type == "image" or item_type == "image-text":
            img_path = ""
            if "|" in item_content:
                text_part, img_part = item_content.split("|", 1)
                img_path = img_part.strip()
                output_html += f"<p><b>Content:</b> {text_part.strip()}</p>"
            else:
                img_path = item_content

            if os.path.exists(img_path):
                base64_image = image_to_base64(img_path)
                if base64_image:
                    output_html += f"<div style='text-align:center;'><img src='{base64_image}' width='500' style='display:inline-block; margin-bottom:10px;'></div>"
                else:
                    output_html += f"<p><i>Could not display image at {img_path}</i></p>"
            else:
                output_html += f"<p><i>Image not found at {img_path}</i></p>"

        else:  # text
            output_html += f"<p><b>Content:</b> {item_content}</p>"

        # --- Display Extracted Info ---
        extracted_info_html = format_extracted_info_html(item.get("extracted_info", "{}"))
        output_html += extracted_info_html

        output_html += (
            f"<p><b>URL:</b> <a href='{item.get('url', '#')}' target='_blank'>{item.get('url', 'N/A')}</a></p>"
        )
        output_html += f"<p><b>Date:</b> {item.get('date', 'N/A')}</p>"
        output_html += "</div>"

    output_html += "</div>"

    # If there are results, show the feedback buttons
    return output_html, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)


# --- Gradio Interface Definition ---

with gr.Blocks() as demo:
    gr.Markdown("# Multimodal Information Retrieval System")

    with gr.Tab("Manage Data"):
        gr.Markdown("## Add New Item to Database")
        with gr.Row():
            with gr.Column():
                add_item_type = gr.Dropdown(["text", "image", "image-text"], label="Item Type")
                add_title = gr.Textbox(label="Title")
                add_content = gr.Textbox(label="Content (Text or Description)")
                add_url = gr.Textbox(label="URL")
                add_image = gr.Image(type="filepath", label="Image (if applicable)")
                add_button = gr.Button("Add Item")
            with gr.Column():
                add_status = gr.Textbox(label="Status")
                add_output = gr.Textbox(label="Added Item Details")

        add_button.click(
            add_item,
            inputs=[add_item_type, add_title, add_content, add_url, add_image],
            outputs=[add_status, add_output, add_title, add_content, add_url, add_image],
        )

    with gr.Tab("Search"):
        gr.Markdown("## Search for Information")

        # --- Initial values for feedback ---
        initial_search_total, initial_search_accurate = load_feedback_scores(SEARCH_FEEDBACK_FILE)
        if initial_search_total > 0:
            initial_search_accuracy_text = f"**Search Accuracy:** {(initial_search_accurate / initial_search_total) * 100:.1f}% ({initial_search_accurate} / {initial_search_total} votes)"
        else:
            initial_search_accuracy_text = "**Search Accuracy:** N/A"

        initial_extraction_total, initial_extraction_accurate = load_feedback_scores(EXTRACTION_FEEDBACK_FILE)
        if initial_extraction_total > 0:
            initial_extraction_accuracy_text = f"**Extraction Accuracy:** {(initial_extraction_accurate / initial_extraction_total) * 100:.1f}% ({initial_extraction_accurate} / {initial_extraction_total} votes)"
        else:
            initial_extraction_accuracy_text = "**Extraction Accuracy:** N/A"

        with gr.Row():
            with gr.Column(scale=2):
                search_query = gr.Textbox(label="Search Query (Text)")
                search_image_query = gr.Image(type="filepath", label="Search Query (Image)")
                search_top_k = gr.Slider(1, 20, value=5, step=1, label="Top K")
                search_button = gr.Button("Search")

                with gr.Column(visible=False) as feedback_column:
                    # --- Search Feedback Section ---
                    gr.Markdown("--- \n ### Was this search result relevant?")
                    with gr.Row():
                        search_accurate_button = gr.Button("👍 Relevant")
                        search_inaccurate_button = gr.Button("👎 Not Relevant")
                    search_accuracy_display = gr.Markdown(initial_search_accuracy_text)

                    # --- Extraction Feedback Section ---
                    gr.Markdown("### Were the extracted details correct?")
                    with gr.Row():
                        extraction_correct_button = gr.Button("👍 Correct")
                        extraction_incorrect_button = gr.Button("👎 Incorrect")
                    extraction_accuracy_display = gr.Markdown(initial_extraction_accuracy_text)

            with gr.Column(scale=3):
                search_results = gr.HTML(label="Search Results")

        # --- State Management ---
        search_total_votes = gr.State(initial_search_total)
        search_accurate_votes = gr.State(initial_search_accurate)
        extraction_total_votes = gr.State(initial_extraction_total)
        extraction_accurate_votes = gr.State(initial_extraction_accurate)

        search_button.click(
            search_items,
            inputs=[search_query, search_image_query, search_top_k],
            outputs=[search_results, feedback_column],  # The outputs list is simplified here for clarity
        )

        # --- Feedback Button Logic ---
        search_accurate_button.click(
            fn=record_search_feedback,
            inputs=[gr.State("accurate"), search_total_votes, search_accurate_votes],
            outputs=[search_total_votes, search_accurate_votes, search_accuracy_display],
        )
        search_inaccurate_button.click(
            fn=record_search_feedback,
            inputs=[gr.State("inaccurate"), search_total_votes, search_accurate_votes],
            outputs=[search_total_votes, search_accurate_votes, search_accuracy_display],
        )
        extraction_correct_button.click(
            fn=record_extraction_feedback,
            inputs=[gr.State("correct"), extraction_total_votes, extraction_accurate_votes],
            outputs=[extraction_total_votes, extraction_accurate_votes, extraction_accuracy_display],
        )
        extraction_incorrect_button.click(
            fn=record_extraction_feedback,
            inputs=[gr.State("incorrect"), extraction_total_votes, extraction_accurate_votes],
            outputs=[extraction_total_votes, extraction_accurate_votes, extraction_accuracy_display],
        )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=10099, allowed_paths=["/"])
