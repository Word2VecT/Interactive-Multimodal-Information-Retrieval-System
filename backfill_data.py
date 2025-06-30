import json
from tqdm import tqdm
import sys
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure the main project directory is in the Python path
# This allows us to import our custom modules like `database` and `openai_extractor`
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import Database
from openai_extractor import extract_information

# --- OpenAI Configuration ---
# You can centralize this config or set it here for the script
# It's better to use environment variables for sensitive data
OPENAI_API_KEY = ""
OPENAI_BASE_URL = ""
OPENAI_MODEL_NAME = "chatgpt-4o-latest"
OPENAI_RETRY_TIMES = 3


def process_item(item_id, metadata, collection):
    """
    Processes a single item: extracts info and updates the database.
    Returns a status tuple: (status_string, item_id, title).
    """
    title = metadata.get("title", f"ID: {item_id}")
    try:
        content = metadata.get("content", "")
        item_type = metadata.get("type", "")
        image_path = None
        text_for_extraction = content

        if item_type == "image":
            image_path = content
            text_for_extraction = title
        elif item_type == "image-text":
            if "|" in content:
                text_part, img_part = content.split("|", 1)
                text_for_extraction = text_part.strip()
                image_path = img_part.strip()
                if not os.path.exists(image_path):
                    image_path = None
            else:
                text_for_extraction = title

        extracted_info = extract_information(
            text=text_for_extraction,
            image_path=image_path,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model_name=OPENAI_MODEL_NAME,
            retry_times=OPENAI_RETRY_TIMES,
        )

        if extracted_info and "error" not in extracted_info:
            new_metadata = metadata.copy()
            new_metadata["extracted_info"] = json.dumps(extracted_info)
            collection.update(ids=[item_id], metadatas=[new_metadata])
            return ("Updated", item_id, title)
        else:
            error_msg = extracted_info.get("error", "Unknown extraction error")
            return ("Failed", item_id, f"{title} - Error: {error_msg}")

    except Exception as e:
        return ("Exception", item_id, f"{title} - Error: {e}")


def backfill_concurrently(force_refresh: bool, max_workers: int):
    """
    Fetches all items and processes them concurrently to backfill extracted information.
    """
    print("Initializing database connection...")
    db = Database()
    collection = db.collection
    print("Database connected.")

    try:
        all_items = collection.get(include=["metadatas"])
        if not all_items or not all_items["ids"]:
            print("Database is empty. Nothing to backfill.")
            return
    except Exception as e:
        print(f"Error fetching data from ChromaDB: {e}", file=sys.stderr)
        return

    items_to_process = []
    if force_refresh:
        print("Force refresh enabled: All items will be re-processed.")
        for item_id, metadata in zip(all_items["ids"], all_items["metadatas"]):
            items_to_process.append((item_id, metadata))
    else:
        print("Standard mode: Only processing items without valid extracted info.")
        for item_id, metadata in zip(all_items["ids"], all_items["metadatas"]):
            if (
                not metadata.get("extracted_info")
                or metadata["extracted_info"] == "{}"
                or "error" in metadata["extracted_info"]
            ):
                items_to_process.append((item_id, metadata))

    if not items_to_process:
        print("No items require processing. Exiting.")
        return

    updated_count = 0
    failed_count = 0

    print(f"Found {len(items_to_process)} items to process. Starting concurrent backfill with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(process_item, item_id, metadata, collection): metadata.get("title")
            for item_id, metadata in items_to_process
        }

        with tqdm(total=len(items_to_process), desc="Processing items") as pbar:
            for future in as_completed(future_to_item):
                status, item_id, details = future.result()
                if status == "Updated":
                    updated_count += 1
                else:
                    failed_count += 1
                    tqdm.write(f"  [!] {status} on item {details}")
                pbar.update(1)
                pbar.set_postfix_str(f"Updated: {updated_count}, Failed: {failed_count}")

    print("\n--- Backfill Summary ---")
    print(f"Total items targeted for processing: {len(items_to_process)}")
    print(f"Successfully updated: {updated_count} items.")
    print(f"Failed or skipped due to errors: {failed_count} items.")
    print("-------------------------")


def main():
    parser = argparse.ArgumentParser(
        description="Concurrently backfill extracted information for items in the database."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="If set, re-processes all items, ignoring existing extracted information.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of concurrent threads to use for processing (default: 4)."
    )
    args = parser.parse_args()

    backfill_concurrently(args.force_refresh, args.workers)


if __name__ == "__main__":
    main()
