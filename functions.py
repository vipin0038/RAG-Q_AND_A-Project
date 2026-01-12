from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from model import hf_emb,processor,model
from pydub import AudioSegment
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders.image import UnstructuredImageLoader

import librosa
import torch
import os


async def split_and_embed(text):

    splitter = RecursiveCharacterTextSplitter(chunk_size = 700,chunk_overlap = 150)
    chunks=   splitter.split_documents(text)
    db =  FAISS.from_documents(chunks,hf_emb)
    retriver =  db.as_retriever(search_type = 'similarity',search_kwargs = {"k":3})
    sparse_retriever = BM25Retriever.from_documents(chunks)

    # Combine both → weights control importance
    hybrid_retriever = EnsembleRetriever(
        retrievers=[retriver, sparse_retriever],
        weights=[0.7, 0.3]  
    )
    return  hybrid_retriever


    # return retriver


def fromat_to_wav(file_location,extention):
        audio = AudioSegment.from_file(file_location,format=extention)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export("output.wav", format='wav')
        os.remove(file_location)
        audio, sr = librosa.load("output.wav", sr=16000)
        return  audio,sr

def convert_audio_to_text(audio,sr) :
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    # Model prediction
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        return transcription


