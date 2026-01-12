from fastapi import FastAPI,Response,Request,UploadFile,File,HTTPException,status
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.templating import Jinja2Templates 
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from functions import convert_audio_to_text,hf_emb,fromat_to_wav
import os
from functions import split_and_embed
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders.image import UnstructuredImageLoader
from dotenv import load_dotenv

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROK_API_KEY not found in .env")

app = FastAPI()

template = Jinja2Templates(directory='templates')

  


llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.2,max_retries=2)

prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer only from the provided context.
If the context is insufficient, just say you don't know.

{context}

Question: {question}
""",input_variables=['context', 'question'])

class question_class(BaseModel):
    question :str 
 

@app.get('/',response_class=HTMLResponse) 
def hello(request:Request):
    return template.TemplateResponse('index.html',{'request':request})


app.state.new_retriver = None

@app.post("/file_submit/")
async def submit_file(file:UploadFile=File(...)) :

    data = []
    try : 
        extension = file.filename.split('.')[-1]
        path = os.getcwd()
        UPLOAD_DIR = "upoad_files"
       # path = r"C:\Users\vipin\Desktop\RAG_Q_A\upoad_files"
        file_location = os.path.join(path,UPLOAD_DIR,file.filename)
        image_extensions = ["jpg", "jpeg", "png" ,"tiff", "tif", "webp"]
        audio_extention = ['mp3', 'wav', 'ogg', 'aac', 'm4a','flac', 'mp4', 'wma', 'aiff', 'alac', 'opus']
        with open(file_location,'wb') as f :
            f.write(await file.read())
    
        
        if extension == 'pdf' :  
            loader = PyPDFLoader(file_location) 
            content =   loader.load()
            retriver =   await split_and_embed(content)
            app.state.new_retriver = retriver
            return {"msg":"sucess"}

            
        elif extension == "txt" :
            with open(file_location,'r') as f :
               text =  f.read()
            text_content = [Document(text)]
            retriver =   await split_and_embed(text_content)
            app.state.new_retriver = retriver
            return {"msg":"sucess"}
  

        elif extension in image_extensions :
            loader  = UnstructuredImageLoader(file_path = file_location)
            document = loader.load()
            retriver =   await split_and_embed(document)
            app.state.new_retriver = retriver
            return {"msg":"sucess"}
            
        elif extension in audio_extention :
            audio,sr =  fromat_to_wav(file_location,extension)
            transcript =  convert_audio_to_text(audio,sr)
            document =  [Document(page_content=transcript[0])]
            retriver =   await split_and_embed(document)
            app.state.new_retriver = retriver
            return {"msg":"sucess"}
            
    except :
       raise HTTPException(detail="File is not in a proper format",status_code=status.HTTP_400_BAD_REQUEST) 
    


@app.post("/get_answer/")
async def question_answer(question:question_class):
    question_h = question.question
    retriver = app.state.new_retriver
    retrived_doc =  retriver.invoke(question_h)
    context_text = "\n\n".join(doc.page_content for doc in retrived_doc)
    final_prompt =  prompt.invoke({"context" : context_text,"question":question_h})
    answer = await llm.ainvoke(final_prompt)
    return {"res":answer.content}
