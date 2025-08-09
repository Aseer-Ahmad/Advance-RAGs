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


class CLIP_RAG:
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
        print(f"loading CLIPMomdel {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        print(f"loading CLIPMomdel processor {model_name}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def generateEmbeddings(self):

        self.embs = []
        self.metadatas = []
        batch_size = 2
        print(f"creating embeddings")
        for i in range(0, len(self.dataset), batch_size):
            batch = self.dataset[i:i+batch_size]
            images = [Image.open(d["image_path"]).convert("RGB") for d in batch]
            texts = [d["metadata_text"] for d in batch]

            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)

                image_emb = out.image_embeds  # (B, D)
                text_emb = out.text_embeds    # (B, D)

                image_emb = F.normalize(image_emb, dim=-1)
                text_emb = F.normalize(text_emb, dim=-1)
                
                print(f"embeddings generated for {i}")
                if self.fuse_type == "hadamard" : 
                    joint = self.fuse_hadamard(image_emb, text_emb)  # (B, D)
                else : 
                    joint = self.fuse_add(image_emb, text_emb)  # (B, D)

                print(f"embeddings fused for {i}")

            self.embs.append(joint.cpu().numpy().astype("float32"))
            for d in batch:
                self.metadatas.append({
                    "image_path": d["image_path"],
                    "metadata_text": d["metadata_text"],
                })

        self.emb_matrix = np.vstack(self.embs)  # (N, D)
        self.createIndex()
        self.save_Index_Metadata()      

    def createIndex(self):
        print("creating index")
        D = self.emb_matrix.shape[1]
        self.index = faiss.IndexFlatIP(D)   
        self.index.add(self.emb_matrix)          

    def save_Index_Metadata(self):
        print("saving index and metadata")
        faiss.write_index(self.index, "joint_index.faiss")
        pd.DataFrame(self.metadatas).to_parquet("metadatas.parquet")

    def fetch_index(self, index_pth, mtd_pth):
        self.index = faiss.read_index(index_pth)
        self.metadf = pd.read_parquet(mtd_pth)

    def embed_query(self, query_str, query_img):
        print("embedding query")
        inputs = self.processor(text=[query_str], images=[query_img], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)
            img_e = F.normalize(out.image_embeds, dim=-1)  # (1, D)
            txt_e = F.normalize(out.text_embeds, dim=-1)   # (1, D)
            if self.fuse_type == "hadamard" : 
                joint = self.fuse_hadamard(img_e, txt_e)                     # (1, D)
            else : 
                joint = self.fuse_add(img_e, txt_e)

        return joint.cpu().numpy().astype("float32")
    
    def retreive(self, query_text, img_pth, k) : 

        print("retreiving")
        query_image = Image.open(img_pth).convert("RGB")
        q_emb = self.embed_query(query_text, query_image)

        scores, indices = self.index.search(q_emb, k)  # scores are inner products
        print(scores)
        print(indices)
        results = self.metadatas[indices[0][0]]
        # results = self.metadf.iloc[indices[0]].to_dict(orient="records")

        print("Top results:", results)
        print("Scores:", scores[0])


if __name__ == '__main__' : 
    
    dataset = [
        {"image_path": "data/imgs/apple.jpg", "metadata_text": "This is apple"},
        {"image_path": "data/imgs/clock.jpg", "metadata_text": "This is clock"},
    ]

    clipindex = CLIP_RAG(dataset, fuse_type="hadamard")
    clipindex.setModel()
    clipindex.generateEmbeddings()

    clipindex.retreive("what is this", "data/imgs/clock3.jpg", k = 1)
