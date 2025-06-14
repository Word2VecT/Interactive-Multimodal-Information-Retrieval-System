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

# --- Feedback Persistence ---
FEEDBACK_FILE = "feedback_scores.json"

def load_feedback_scores():
    """Loads feedback scores from a JSON file."""
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            try:
                scores = json.load(f)
                return scores.get('total', 0), scores.get('accurate', 0)
            except json.JSONDecodeError:
                return 0, 0
    return 0, 0

def save_feedback_scores(total, accurate):
    """Saves feedback scores to a JSON file."""
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump({'total': total, 'accurate': accurate}, f)

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
if not os.path.exists('data'):
    os.makedirs('data')

db = Database()
retriever = Retriever()

# --- Functions for Gradio Interface ---

def add_item(item_type, title, content, url, image_path):
    """Adds a new item to the database after generating its embedding."""
    try:
        print(f"--- Received request to add item of type: {item_type} ---")
        if not title or (item_type in ["text", "image-text"] and not content):
            error_msg = "Title and Content are required for this item type."
            print(f"Validation failed: {error_msg}")
            # On validation failure, keep inputs for user to correct
            return error_msg, error_msg, title, content, url, image_path

        embedding = None
        saved_image_path = None
        final_content = content

        if item_type == "text":
            print(f"Generating text embedding for title: {title}")
            embedding = retriever.get_text_embedding(content)
            print("Text embedding generated successfully.")
        
        elif item_type == "image":
            if image_path is None:
                error_msg = "Image is required for item type 'image'."
                print(f"Validation failed: {error_msg}")
                return error_msg, error_msg, title, content, url, image_path
            saved_image_path = os.path.join('data', os.path.basename(image_path))
            shutil.copy(image_path, saved_image_path)
            final_content = saved_image_path
            print(f"Generating image embedding for image: {saved_image_path}")
            embedding = retriever.get_image_embedding(saved_image_path)
            print("Image embedding generated successfully.")

        elif item_type == "image-text":
            if image_path is None or not content:
                error_msg = "Image and Content are required for item type 'image-text'."
                print(f"Validation failed: {error_msg}")
                return error_msg, error_msg, title, content, url, image_path
            saved_image_path = os.path.join('data', os.path.basename(image_path))
            shutil.copy(image_path, saved_image_path)
            final_content = f"{content} | {saved_image_path}"
            print(f"Generating image-text embedding for: {final_content}")
            embedding = retriever.get_image_text_embedding(saved_image_path, content)
            print("Image-text embedding generated successfully.")

        if embedding is not None:
            current_date = datetime.now().isoformat()
            print(f"Adding item to database...")
            db.add(item_type, title, final_content, url, current_date, embedding)
            print("Item added to database successfully.")
            added_details = f"Type: {item_type}\\nTitle: {title}\\nContent: {final_content}\\nURL: {url}\\nDate: {current_date}"
            # On success, clear all input fields for the next entry
            return f"'{title}' added successfully!", added_details, "", "", "", None
        
        # This case might happen if embedding generation fails unexpectedly
        print("Failed to add item because embedding was not generated.")
        return "Failed to add item. Check logs.", "", title, content, url, image_path

    except Exception as e:
        print("!!!!!!!! AN ERROR OCCURRED IN add_item !!!!!!!!")
        print(traceback.format_exc())
        error_message = f"An error occurred: {e}"
        # On exception, keep inputs for user to retry
        return error_message, traceback.format_exc(), title, content, url, image_path

def record_feedback(choice, total, accurate):
    """Updates the accuracy score and saves it."""
    total += 1
    if choice == "accurate":
        accurate += 1
    
    save_feedback_scores(total, accurate)

    if total > 0:
        accuracy_percent = (accurate / total) * 100
        accuracy_text = f"**Accuracy:** {accuracy_percent:.1f}% ({accurate} / {total} votes)"
    else:
        accuracy_text = "**Accuracy:** N/A"
        
    return total, accurate, accuracy_text

def search_items(query, image_query_path, top_k):
    """Performs a search and returns formatted results."""
    # Convert Gradio temp path to a usable path
    temp_image_path = image_query_path # gr.Image with type="filepath" gives a string path

    results = retriever.search(query, temp_image_path, int(top_k))
    
    if not results:
        return "<p>No results found.</p>"

    output_html = "<div>"
    for sim, item in results:
        output_html += "<div style='border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 5px;'>"
        output_html += f"<h3>{item.get('title', 'N/A')} (Score: {sim:.4f})</h3>"
        
        item_content = item.get('content', '')
        item_type = item.get('type', '')

        if item_type == 'image' or item_type == 'image-text':
            img_path = ''
            if '|' in item_content:
                text_part, img_part = item_content.split('|', 1)
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

        else: # text
            output_html += f"<p><b>Content:</b> {item_content}</p>"

        output_html += f"<p><b>URL:</b> <a href='{item.get('url', '#')}' target='_blank'>{item.get('url', 'N/A')}</a></p>"
        output_html += f"<p><b>Date:</b> {item.get('date', 'N/A')}</p>"
        output_html += "</div>"
    
    output_html += "</div>"
    return output_html

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
            outputs=[add_status, add_output, add_title, add_content, add_url, add_image]
        )

    with gr.Tab("Search"):
        gr.Markdown("## Search for Information")
        # --- Initial values for feedback ---
        initial_total, initial_accurate = load_feedback_scores()
        if initial_total > 0:
            initial_accuracy_text = f"**Accuracy:** {(initial_accurate / initial_total) * 100:.1f}% ({initial_accurate} / {initial_total} votes)"
        else:
            initial_accuracy_text = "**Accuracy:** N/A"

        with gr.Row():
            with gr.Column():
                search_query = gr.Textbox(label="Search Query (Text)")
                search_image_query = gr.Image(type="filepath", label="Search Query (Image)")
                search_top_k = gr.Slider(1, 20, value=5, step=1, label="Top K")
                search_button = gr.Button("Search")
                  
                # --- Feedback Section ---
                gr.Markdown("### Was this search result accurate?")
                with gr.Row():
                    accurate_button = gr.Button("👍 Accurate")
                    inaccurate_button = gr.Button("👎 Inaccurate")
                
                accuracy_display = gr.Markdown(initial_accuracy_text)

            with gr.Column():
                search_results = gr.HTML(label="Search Results")

        # --- State for Feedback ---
        total_votes = gr.State(initial_total)
        accurate_votes = gr.State(initial_accurate)

        search_button.click(
            search_items,
            inputs=[search_query, search_image_query, search_top_k],
            outputs=search_results
        )

        # --- Feedback Button Logic ---
        accurate_button.click(
            fn=record_feedback,
            inputs=[gr.State("accurate"), total_votes, accurate_votes],
            outputs=[total_votes, accurate_votes, accuracy_display]
        )

        inaccurate_button.click(
            fn=record_feedback,
            inputs=[gr.State("inaccurate"), total_votes, accurate_votes],
            outputs=[total_votes, accurate_votes, accuracy_display]
        )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=10099, allowed_paths=["/"]) 