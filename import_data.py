import json
import argparse
from retriever import Retriever
from datetime import datetime
from tqdm import tqdm
import sys

def import_from_json(input_file, output_file):
    """
    Imports data from a JSON file into the retrieval system's database.

    Args:
        input_file (str): Path to the input JSON file (e.g., 'data.json').
        output_file (str): Path to the output JSON file to save imported records.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'", file=sys.stderr)
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file}'. Please check its format.", file=sys.stderr)
        return

    print("Initializing retriever... (This may take a moment to load the model)")
    retriever = Retriever()
    db = retriever.db
    print("Retriever initialized.")

    successfully_imported = []
    skipped_count = 0

    print(f"Starting import from '{input_file}'...")
    for record in tqdm(data, desc="Importing records"):
        # Validate that all required fields are present and not empty
        title = record.get("title")
        abstract = record.get("abstract")
        url = record.get("URL")

        if not all([title, abstract, url]):
            skipped_count += 1
            continue

        try:
            # 1. Generate the document embedding (is_query defaults to False)
            embedding = retriever.get_text_embedding(abstract)

            # 2. Add the item to the database
            item_id = db.add(
                item_type='text',
                title=title,
                content=abstract,
                url=url,
                date=datetime.now().isoformat(),
                embedding=embedding
            )

            # 3. Prepare the record for the output file
            imported_record = {
                'id_in_db': item_id,
                'title': title,
                'content': abstract,
                'url': url
            }
            successfully_imported.append(imported_record)

        except Exception as e:
            print(f"\nAn error occurred while processing record titled '{title}': {e}", file=sys.stderr)
            skipped_count += 1
            continue

    # Save the successfully imported records to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(successfully_imported, f, indent=4, ensure_ascii=False)

    print("\n--- Import Summary ---")
    print(f"Successfully imported: {len(successfully_imported)} records.")
    print(f"Skipped (missing fields or error): {skipped_count} records.")
    print(f"Results of imported records saved to '{output_file}'.")
    print("----------------------")

def main():
    parser = argparse.ArgumentParser(description="Bulk import data from a JSON file.")
    parser.add_argument(
        'input_file', 
        default='data.json',
        nargs='?', # Makes the argument optional
        help="Path to the input JSON file (default: 'data.json')."
    )
    parser.add_argument(
        '--output', 
        default='imported_data.json', 
        help="Path for the output JSON file (default: 'imported_data.json')."
    )
    args = parser.parse_args()

    import_from_json(args.input_file, args.output)

if __name__ == '__main__':
    main() 