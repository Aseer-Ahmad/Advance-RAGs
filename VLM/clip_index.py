import os
import json
import torch
import numpy as np
import faiss
import pandas as pd
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)

# helper to fuse embeddings (choose one)
def fuse_hadamard(image_emb, text_emb):
    # both should be L2-normalized already; compute elementwise product then renormalize
    fused = image_emb * text_emb
    return F.normalize(fused, dim=-1)

def fuse_add(image_emb, text_emb):
    fused = image_emb + text_emb
    return F.normalize(fused, dim=-1)

# pick fusion
fuse = fuse_hadamard

# 2) Build index from dataset
# dataset: list of dicts with keys: image_path, metadata_text (labels / captions / class)
dataset = [
    # example entries
    # {"image_path": "data/img1.jpg", "metadata_text": "dog, brown, running"},
    # ...
]

# load/generate dataset list (replace with your loader)
# dataset = load_my_dataset()

batch_size = 32
embs = []
metadatas = []

for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i+batch_size]
    images = [Image.open(d["image_path"]).convert("RGB") for d in batch]
    texts = [d["metadata_text"] for d in batch]

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
        # CLIPModel returns text_embeds and image_embeds
        image_emb = out.image_embeds  # (B, D)
        text_emb = out.text_embeds    # (B, D)

        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        joint = fuse(image_emb, text_emb)  # (B, D)

    embs.append(joint.cpu().numpy().astype("float32"))
    for d in batch:
        metadatas.append({
            "image_path": d["image_path"],
            "metadata_text": d["metadata_text"],
            # add any labels / ids / more fields you want
        })

emb_matrix = np.vstack(embs)  # (N, D)

# 3) Create FAISS index (cosine via inner product on normalized vectors)
D = emb_matrix.shape[1]
index = faiss.IndexFlatIP(D)   # simple exact index
index.add(emb_matrix)          # add vectors

# persist index + metadata
faiss.write_index(index, "joint_index.faiss")
pd.DataFrame(metadatas).to_parquet("metadatas.parquet")



class CLIP_index:
    def __init__(self, dataset, fuse_type = "hadamard") :
        self.fuse_type = fuse_type
        self.dataset = dataset

    def fuse_hadamard(self, image_emb, text_emb):
        fused = image_emb * text_emb
        return F.normalize(fused, dim=-1)

    def fuse_add(self, image_emb, text_emb):
        fused = image_emb + text_emb
        return F.normalize(fused, dim=-1)
            
    def setModel(self) : 
        model_name = "openai/clip-vit-base-patch32"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def generateEmbeddings(self):

        self.embs = []
        self.metadatas = []

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            images = [Image.open(d["image_path"]).convert("RGB") for d in batch]
            texts = [d["metadata_text"] for d in batch]

            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model(**inputs)

                image_emb = out.image_embeds  # (B, D)
                text_emb = out.text_embeds    # (B, D)

                image_emb = F.normalize(image_emb, dim=-1)
                text_emb = F.normalize(text_emb, dim=-1)
                
                if self.fuse_type == "hadamard" : 
                    joint = self.fuse_hadamard(image_emb, text_emb)  # (B, D)
                else : 
                    joint = self.fuse_add(image_emb, text_emb)  # (B, D)

            embs.append(joint.cpu().numpy().astype("float32"))
            for d in batch:
                metadatas.append({
                    "image_path": d["image_path"],
                    "metadata_text": d["metadata_text"],
                })

        self.emb_matrix = np.vstack(embs)  # (N, D)            

    def createIndex(self):
        D = emb_matrix.shape[1]
        self.index = faiss.IndexFlatIP(D)   
        self.index.add(emb_matrix)          


    def save_Index_Metadata(self):
        faiss.write_index(self.index, "joint_index.faiss")
        pd.DataFrame(metadatas).to_parquet("metadatas.parquet")

