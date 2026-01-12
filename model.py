from langchain_huggingface import HuggingFaceEmbeddings
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


hf_emb = HuggingFaceEmbeddings(model =  "sentence-transformers/all-MiniLM-L6-v2")


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
