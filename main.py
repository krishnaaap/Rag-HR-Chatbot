from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone, ServerlessSpec

from utils_phase3 import handle_leave_flow, handle_multi_intent_flow
from utils_email import send_leave_email

# ------------------- FastAPI Setup -------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- User sessions -------------------
user_sessions = {}
warnings_file = "user_warnings.json"
if os.path.exists(warnings_file):
    with open(warnings_file, "r") as f:
        user_warnings = json.load(f)
else:
    user_warnings = {}

def save_warnings():
    with open(warnings_file, "w") as f:
        json.dump(user_warnings, f)

# ------------------- Config -------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_API_KEY_HERE")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_API_KEY_HERE")
PINECONE_INDEX_NAME = "rag-fastapi"

# ------------------- Pinecone + RAG Setup -------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(PINECONE_INDEX_NAME)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index=index, embedding=embedding, text_key="text")

llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192", temperature=0.3)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    memory=memory
)

# ------------------- ML Models -------------------
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")
intent_tokenizer = AutoTokenizer.from_pretrained("./multi_intent_bert")
intent_model = AutoModelForSequenceClassification.from_pretrained("./multi_intent_bert")
INTENT_LABELS = ["policy related", "leave intension", "greeting"]
nlp_ner = spacy.load("./ner_model_15lpa")

# ------------------- Helpers -------------------
def detect_intents(text):
    inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = intent_model(**inputs)
    probs = torch.sigmoid(outputs.logits).numpy()[0]
    detected = []
    for i, p in enumerate(probs):
        if p >= 0.5:
            detected.append((INTENT_LABELS[i], float(p)))
    return detected

def extract_entities(text):
    doc = nlp_ner(text)
    return [{"entity_group": ent.label_, "word": ent.text} for ent in doc.ents]

def get_rag_answer(query: str) -> str:
    try:
        return qa_chain.run(query)
    except Exception as e:
        print("RAG retrieval error:", e)
        return "Sorry, I could not find relevant information."

# ------------------- Request Model -------------------
class QueryRequest(BaseModel):
    user_id: str
    query: str

# ------------------- Main Endpoint -------------------
@app.post("/ask")
async def ask_question(request: QueryRequest):
    user_id = request.user_id
    query_raw = request.query.strip()

    # Block toxic users
    if user_warnings.get(user_id, 0) >= 2:
        return {"warning": "Access restricted due to repeated toxic behavior."}

    toxic_result = toxicity_classifier(query_raw)[0]
    if toxic_result["label"].lower() == "toxic" and toxic_result["score"] > 0.6:
        user_warnings[user_id] = user_warnings.get(user_id, 0) + 1
        save_warnings()
        return {"warning": "Please keep respectful language."}

    # Continue leave flow if session exists
    if user_id in user_sessions:
        entities = extract_entities(query_raw)
        hr_response = handle_leave_flow(
            user_id=user_id,
            query_raw=query_raw,
            entities=entities,
            conversation_states=user_sessions,
            send_leave_email=send_leave_email,
            llm=llm
        )
        return {"response": hr_response}

    # Main multi-intent processing
    entities = extract_entities(query_raw)
    hr_response = handle_multi_intent_flow(
        user_id=user_id,
        query_raw=query_raw,
        entities=entities,
        conversation_states=user_sessions,
        send_leave_email=send_leave_email,
        get_rag_answer=get_rag_answer,
        llm=llm,
        bert_classifier=detect_intents,
        ner_model=extract_entities,
        toxicity_classifier=toxicity_classifier
    )

    if hr_response:
        return {"response": hr_response}

    # Fallback
    return {"response": "I can help with HR topics like leave or policies. Could you rephrase?"}
