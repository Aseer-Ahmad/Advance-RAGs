from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

documents = [
    "This is a list which containing sample documents.",
    "Keywords are important for keyword-based search.",
    "Document analysis involves extracting keywords.",
    "Keyword-based search relies on sparse embeddings.",
    "Understanding document structure aids in keyword extraction.",
    "Efficient keyword extraction enhances search accuracy.",
    "Semantic similarity improves document retrieval performance.",
    "Machine learning algorithms can optimize keyword extraction methods."
]

class Reranking : 
    def __init__(self) : 
        model_name = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, documents):
        self.doc_embeddings  = self.model.encode(documents, convert_to_tensor=True)

    def get_embedding_len(self) : 
        return len(self.doc_embeddings[0])
    
    def match_documents(self, query):
        query_emdding = self.model.encode(query, convert_to_tensor=True)
        similarities = cosine_similarity(np.array([query_emdding]), np.array(self.doc_embeddings))
        most_sim_indx = np.argmax(similarities)


import weaviate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from transformers import ( AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, )
from langchain import HuggingFacePipeline
import torch

class Weavite_Hybrid :
    def _init__(self):
        pass

    def get_client(self, WEAVIATE_URL, WEAVIATE_API_KEY, HF_TOKEN):
        self.client = weaviate.Client(
            url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY),
            additional_headers={
                "X-HuggingFace-Api-Key": HF_TOKEN
            },
        )

    def weavite_retreiver(self) : 

        self.retriever = WeaviateHybridSearchRetriever(
            alpha = 0.5,               # defaults to 0.5, which is equal weighting between keyword and semantic search
            client = self.client,           # keyword arguments to pass to the Weaviate client
            index_name = "RAG",  # The name of the index to use
            text_key = "content",         # The name of the text key to use
            attributes = [], # The attributes to return in the results
            create_schema_if_missing=True,
        )

    def get_model_pipeline(self):

        model_name = "HuggingFaceH4/zephyr-7b-beta"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
        self.tokenizer.bos_token_id = 1

        self.pipeline = HuggingFacePipeline(pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            #max_length=2048,
            do_sample=True,
            top_k=5,
            max_new_tokens=100,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        ))


    def langchain_PDF_loader(self, doc_pth):
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(doc_pth)
        self.docs = loader.load()
        