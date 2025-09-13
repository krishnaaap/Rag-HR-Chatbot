# Rag-HR-Chatbot
AI-Powered HR Chatbot  Built an HR assistant using FastAPI, React, and Groq LLM for leave and policy queries.  Implemented multi-intent detection with BERT and RAG-based retrieval for accurate responses.  Extracted leave dates &amp; reasons using NER and automated HR email notifications.  Developed a user-friendly React UI 


# HR Assistant Chatbot

A **Generative AI-powered HR Assistant** built with FastAPI, Transformers (BERT/DistilBERT), LangChain, Pinecone, and a React.js frontend.  
This chatbot automates HR-related queries, such as leave requests, policy questions, and general greetings.

---

## Features

- **Multi-intent Classification:** Detects leave requests, policy-related questions, and greetings using a BERT-based model.
- **RAG Pipeline:** Uses LangChain + Pinecone for retrieving relevant HR documents and answering queries.
- **Leave Management:** Extracts leave dates, validates against weekends/holidays, and sends emails to HR.
- **Toxicity Detection:** Filters inappropriate user inputs with a pre-trained model.
- **Interactive React.js UI:** Clean chat interface for users to interact with the bot.

---

## Tech Stack

- **Backend:** FastAPI, Python
- **AI/ML:** Transformers (HuggingFace), PyTorch, LangChain, Pinecone
- **Frontend:** React.js, CSS
- **Email:** SMTP via Gmail
- **Vector Database:** Pinecone for RAG retrieval
- **NLP Models:** Multi-intent BERT, NER via spaCy, Toxicity classification

---

