from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class Input_sentence(BaseModel):
    input_sentence: str

@app.post('/request')
async def predict(input_sentence: Input_sentence):
    print('요청받음')
    value = input_sentence.input_sentence
    print(value)
    return JSONResponse(content={'sentene': value})