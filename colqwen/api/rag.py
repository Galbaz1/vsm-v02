"""Multimodal RAG for PDF retrieval using ColQwen2.5 and Qwen2.5-VL"""

import os
import warnings
import torch
from dotenv import load_dotenv
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from qwen_vl_utils import process_vision_info
import weaviate
from weaviate.classes.query import MetadataQuery

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*Qwen2VLImageProcessor.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="uvicorn.*")

load_dotenv()


class ColVisionEmbedder:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def multi_vectorize_text(self, query):
        batch = self.processor.process_queries([query]).to(self.model.device)
        with torch.no_grad():
            embedding = self.model(**batch)
        return embedding[0]


class QwenVLModel:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
    
    def query_images(self, query, images):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                *[{"type": "image", "image": img} for img in images]
            ]
        }]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]


class MultimodalRAG:
    def __init__(self, embedder, vlm, weaviate_client, collection_name="PDFDocuments"):
        self.embedder = embedder
        self.vlm = vlm
        self.client = weaviate_client
        self.collection = weaviate_client.collections.get(collection_name)
    
    def retrieve(self, query, top_k=3):
        query_embedding = self.embedder.multi_vectorize_text(query)
        response = self.collection.query.near_vector(
            near_vector=query_embedding.cpu().numpy(),
            limit=top_k,
            return_metadata=MetadataQuery(distance=True)
        )
        return response.objects
    
    def generate_answer(self, query, retrieved_objects):
        if not retrieved_objects:
            return "No relevant documents found."
        
        context = "\\n\\n".join([
            f"Page {obj.properties['page_number']} from {obj.properties['asset_manual']}"
            for obj in retrieved_objects
        ])
        
        prompt = f"""Based on the following document pages, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        return self.vlm.query_images(prompt, [])
    
    def chat(self, query, top_k=3):
        retrieved = self.retrieve(query, top_k)
        answer = self.generate_answer(query, retrieved)
        
        return {
            "answer": answer,
            "retrieved_pages": [
                {
                    "page_id": obj.properties["page_id"],
                    "asset_manual": obj.properties["asset_manual"],
                    "page_number": obj.properties["page_number"],
                    "distance": obj.metadata.distance
                }
                for obj in retrieved
            ]
        }


def _initialize():
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    attn = "flash_attention_2" if is_flash_attn_2_available() else "eager"
    
    print(f"Device: {device}, Attention: {attn}")
    
    print("Loading ColQwen2.5...")
    colqwen_model = ColQwen2_5.from_pretrained(
        "vidore/colqwen2.5-v0.2",
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation=attn,
    ).eval()
    colqwen_processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")
    embedder = ColVisionEmbedder(colqwen_model, colqwen_processor)
    
    print("Loading Qwen2.5-VL...")
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation=attn,
    ).eval()
    qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    vlm = QwenVLModel(qwen_model, qwen_processor, device)
    
    wcd_url = os.environ.get("WEAVIATE_URL")
    wcd_key = os.environ.get("WEAVIATE_API_KEY")
    
    if not wcd_url or not wcd_key:
        raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY required in .env")
    
    print("Connecting to Weaviate...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=wcd_url,
        auth_credentials=weaviate.auth.AuthApiKey(wcd_key),
    )
    
    print("RAG system ready!")
    return MultimodalRAG(embedder, vlm, client)


rag = _initialize()


def query(text: str, top_k: int = 3) -> dict:
    """Query RAG system with retrieval and answer generation"""
    return rag.chat(text, top_k)


def retrieve_only(text: str, top_k: int = 3) -> list:
    """Retrieve relevant pages without answer generation"""
    retrieved = rag.retrieve(text, top_k)
    return [
        {
            "page_id": obj.properties["page_id"],
            "asset_manual": obj.properties["asset_manual"],
            "page_number": obj.properties["page_number"],
            "distance": obj.metadata.distance
        }
        for obj in retrieved
    ]
