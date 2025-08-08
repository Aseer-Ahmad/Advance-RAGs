from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever

documents = [
    "This is a list which containig sample documents.",
    "Keywords are important for keyword-based search.",
    "Document analysis involves extracting keywords.",
    "Keyword-based search relies on sparse embeddings."
]

class HybridRetriever : 
    def __init__(self) : 
        self.vectorizer = TfidfVectorizer()

    def vectorize_document(self, document):
        return self.vectorizer.fit_transform([document])

    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def get_embedding_model(self, HF_TOKEN):
        from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
        embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5")
        return embeddings
    
    def get_retreivers(self, documents):
        chunks = self.split_documents(documents)
        embeddings = self.get_embedding_model(HF_TOKEN="your_huggingface_token_here")

        vectorstore=Chroma.from_documents(chunks,embeddings)
        vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": 3})

        keyword_retriever = BM25Retriever.from_documents(chunks)
        keyword_retriever.k =  3
        ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,keyword_retriever],weights=[0.3, 0.7])

        return vectorstore_retreiver, ensemble_retriever
    
    def retreival_chains(self, llm, documents):
        from langchain.chains import RetrievalQA

        vectorstore_retreiver, ensemble_retriever = self.get_retreivers(documents)

        normal_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore_retreiver
        )

        hybrid_chain  = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=ensemble_retriever
        )
