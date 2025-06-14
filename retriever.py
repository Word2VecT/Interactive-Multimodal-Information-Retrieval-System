import torch
from modeling_gme_qwen2vl import GmeQwen2VL
from PIL import Image
import numpy as np
from database import Database
import io
import base64

class Retriever:
    def __init__(self, model_name='Alibaba-NLP/gme-Qwen2-VL-7B-Instruct', db_path='database.json'):
        print("Loading model... This may take a while.")
        self.model = GmeQwen2VL.from_pretrained(
            "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
            torch_dtype="float16", device_map='cuda', trust_remote_code=True
        )
        self.db = Database(path="./database")
        self.search_instruction = 'Find a document that matches the given query.'
        print("Model loaded successfully.")

    def get_text_embedding(self, text, is_query=False, instruction=None):
        with torch.no_grad():
            embedding_tensor = self.model.get_text_embeddings(texts=[text], is_query=is_query, instruction=instruction)
            return embedding_tensor[0].cpu().numpy()

    def get_image_embedding(self, image_path, is_query=False, instruction=None):
        with torch.no_grad():
            embedding_tensor = self.model.get_image_embeddings(images=[image_path], is_query=is_query, instruction=instruction)
            return embedding_tensor[0].cpu().numpy()

    def get_image_text_embedding(self, image_path, text, is_query=False, instruction=None):
        with torch.no_grad():
            embedding_tensor = self.model.get_fused_embeddings(texts=[text], images=[image_path], is_query=is_query, instruction=instruction)
            return embedding_tensor[0].cpu().numpy()

    def _cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def search(self, query, image_query_path=None, top_k=5):
        query_embedding = None
        if image_query_path:
            if query:  # image-text query
                query_embedding = self.get_image_text_embedding(image_query_path, query, is_query=True, instruction=self.search_instruction)
            else:  # image query
                query_embedding = self.get_image_embedding(image_query_path, is_query=True, instruction=self.search_instruction)
        elif query:  # text query
            query_embedding = self.get_text_embedding(query, is_query=True, instruction=self.search_instruction)
        else:
            return [] # No query provided

        if query_embedding is None:
            return []

        results = self.db.query(query_embedding, top_k=top_k)

        # Format results to be consistent with the old structure: (similarity, item_dict)
        formatted_results = []
        if results and results['ids'][0]:
            ids = results['ids'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for i in range(len(ids)):
                similarity = 1 - distances[i]
                item = metadatas[i]
                item['id'] = ids[i]
                formatted_results.append((similarity, item))

        return formatted_results 