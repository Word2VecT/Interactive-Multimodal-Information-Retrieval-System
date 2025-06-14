import argparse
from retriever import Retriever
from datetime import datetime
import os
import sys

def handle_add(args):
    """Handles the 'add' command."""
    print("Initializing retriever for adding item...")
    retriever = Retriever()
    db = retriever.db

    item_type = args.type
    title = args.title
    content = args.content
    url = args.url
    image_path = args.image_path

    embedding = None
    final_content = content

    print(f"Processing item type: {item_type}")
    if item_type == "text":
        if not content:
            print("Error: --content is required for type 'text'", file=sys.stderr)
            return
        print(f"Generating text embedding for: {title}")
        embedding = retriever.get_text_embedding(content)
        final_content = content

    elif item_type == "image":
        if not image_path:
            print("Error: --image_path is required for type 'image'", file=sys.stderr)
            return
        if not os.path.exists(image_path):
            print(f"Error: Image path does not exist: {image_path}", file=sys.stderr)
            return
        print(f"Generating image embedding for: {image_path}")
        embedding = retriever.get_image_embedding(image_path)
        final_content = image_path 

    elif item_type == "image-text":
        if not image_path or not content:
            print("Error: --image_path and --content are required for type 'image-text'", file=sys.stderr)
            return
        if not os.path.exists(image_path):
            print(f"Error: Image path does not exist: {image_path}", file=sys.stderr)
            return
        print(f"Generating image-text embedding for: {content} | {image_path}")
        embedding = retriever.get_image_text_embedding(image_path, content)
        final_content = f"{content} | {image_path}"
    
    else:
        print(f"Error: Unknown item type '{item_type}'", file=sys.stderr)
        return

    if embedding is not None:
        current_date = datetime.now().isoformat()
        db.add(item_type, title, final_content, url, current_date, embedding)
        print(f"\\nSuccessfully added '{title}' to the database.")
    else:
        print("\\nFailed to add item.", file=sys.stderr)


def handle_search(args):
    """Handles the 'search' command."""
    print("Initializing retriever for searching...")
    retriever = Retriever()

    query = args.query
    image_query_path = args.image_query_path
    top_k = args.top_k

    if not query and not image_query_path:
        print("Error: At least one of --query or --image_query_path is required for searching.", file=sys.stderr)
        return
        
    if image_query_path and not os.path.exists(image_query_path):
        print(f"Error: Image path does not exist: {image_query_path}", file=sys.stderr)
        return

    print(f"\nSearching for query: '{query}' with image: '{image_query_path}' (top_k={top_k})...")
    results = retriever.search(query, image_query_path, top_k)

    if not results:
        print("\nNo results found.")
        return

    print("\n--- Search Results ---")
    for i, (sim, item) in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Similarity: {sim:.4f}")
        print(f"  Title: {item.get('title', 'N/A')}")
        print(f"  Type: {item.get('type', 'N/A')}")
        print(f"  Content: {item.get('content', 'N/A')}")
        print(f"  URL: {item.get('url', 'N/A')}")
        print(f"  Date: {item.get('date', 'N/A')}")
    print("----------------------")


def main():
    """Main function to parse arguments and call handlers."""
    parser = argparse.ArgumentParser(description="Command-line interface for the Multimodal Retrieval System.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- Add command ---
    parser_add = subparsers.add_parser('add', help='Add an item to the database')
    parser_add.add_argument('type', choices=['text', 'image', 'image-text'], help='The type of item to add.')
    parser_add.add_argument('--title', required=True, help='Title of the item.')
    parser_add.add_argument('--content', help='Text content or description.')
    parser_add.add_argument('--url', default='', help='URL associated with the item.')
    parser_add.add_argument('--image_path', help='Path to the image file.')
    parser_add.set_defaults(func=handle_add)

    # --- Search command ---
    parser_search = subparsers.add_parser('search', help='Search for items in the database')
    parser_search.add_argument('--query', help='The text query to search for.')
    parser_search.add_argument('--image_query_path', help='Path to an image for the query.')
    parser_search.add_argument('--top_k', type=int, default=5, help='Number of top results to return.')
    parser_search.set_defaults(func=handle_search)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main() 