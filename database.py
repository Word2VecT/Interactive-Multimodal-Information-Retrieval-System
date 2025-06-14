import chromadb
import uuid
import numpy as np

class Database:
    def __init__(self, path="./database", collection_name="retrieval_collection"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, item_type, title, content, url, date, embedding):
        item_id = str(uuid.uuid4())
        metadatas = {
            'type': item_type,
            'title': title,
            'content': content,
            'url': url,
            'date': date
        }
        
        self.collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[metadatas],
            ids=[item_id]
        )
        return item_id

    def query(self, query_embedding, top_k=5):
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        return results

    def get_item_by_id(self, item_id):
        return self.collection.get(ids=[item_id])