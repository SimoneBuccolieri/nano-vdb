import json
import urllib.request
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import nano_vdb

app = FastAPI(title="NanoVDB Microservice")

# Permettiamo CORS (utile se volessimo chiamare direttamente da browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Carichiamo il modello AI di embedding
print("Caricamento del modello SentenceTransformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')
DIMS = model.get_embedding_dimension()

# 2. Inizializziamo il database vettoriale in C++
print(f"Inizializzazione NanoVectorDB con {DIMS} dimensioni...")
db = nano_vdb.NanoVectorDB(DIMS)

# 3. Manteniamo una mappa in memoria per associare l'ID del vettore al testo originale
# Nelle applicazioni reali, questo sarebbe memorizzato in un database SQL/chiave-valore classico.
document_store = {}

# Inseriamo dei documenti di esempio di default per non partire a vuoto
default_docs = [
    "Roma è la capitale dell'Italia e ospita il famoso Colosseo.",
    "La pizza Margherita è un piatto tipico nato a Napoli nel 1889.",
    "Il Giappone è una nazione insulare situata nell'Oceano Pacifico.",
    "L'intelligenza artificiale generativa e gli LLM stanno rivoluzionando la tecnologia.",
    "Il Monte Bianco è la montagna più alta della catena delle Alpi.",
    "Steve Jobs e Steve Wozniak hanno fondato la Apple nel garage dei loro genitori."
]

print("Popolamento del database con i documenti di default...")
embeddings = model.encode(default_docs)
for i, (testo, vettore) in enumerate(zip(default_docs, embeddings)):
    doc_id = f"doc_{i}"
    db.insert(doc_id, vettore.tolist())
    document_store[doc_id] = testo


# Modelli Pydantic per validare l'input delle API
class InsertRequest(BaseModel):
    id: str
    text: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    mode: str = "graph"

class RagRequest(BaseModel):
    query: str


@app.get("/status")
def get_status():
    return {
        "dimensions": DIMS,
        "size": db.size(),
        "documents": list(document_store.values())
    }

@app.post("/insert")
def insert_document(req: InsertRequest):
    try:
        # Generiamo il vettore embedding per il testo in ingresso
        vector = model.encode(req.text).tolist()
        # Inseriamo nel database C++ (che esegue la quantizzazione SQ8 internamente!)
        success = db.insert(req.id, vector)
        if not success:
            raise HTTPException(status_code=500, detail="Errore nell'inserimento del vettore nel database C++")
        
        # Salviamo la mappa del testo
        document_store[req.id] = req.text
        return {"status": "success", "id": req.id, "text": req.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_documents(req: SearchRequest):
    try:
        # Generiamo il vettore embedding per la query
        query_vector = model.encode(req.query).tolist()
        # Eseguiamo la ricerca sul grafo C++ quantizzato o scansione lineare
        if req.mode == "linear":
            results = db.search_linear(query_vector, req.top_k)
        else:
            results = db.search_graph(query_vector, req.top_k)
        
        formatted_results = []
        for r in results:
            text = document_store.get(r.id, "Testo non trovato")
            formatted_results.append({
                "id": r.id,
                "score": round(r.score, 4),
                "text": text
            })
        return {"results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag")
def run_rag(req: RagRequest):
    try:
        # 1. Recupero Semantico dal DB C++
        query_vector = model.encode(req.query).tolist()
        results = db.search_graph(query_vector, top_k=1)
        
        if not results:
            return {"query": req.query, "context": "", "response": "Nessun documento trovato nel database."}
        
        vincitore = results[0]
        contesto_trovato = document_store.get(vincitore.id, "")
        
        # 2. Generazione con LLM via Ollama
        prompt = f"""Sei un assistente intelligente. Usa SOLO le informazioni fornite nel seguente contesto per rispondere alla domanda dell'utente. Se l'informazione non è nel contesto, rispondi 'Non lo so'.

CONTESTO:
{contesto_trovato}

DOMANDA DELL'UTENTE:
{req.query}

RISPOSTA BREVE IN ITALIANO:
"""
        data = {
            "model": "phi3:mini",
            "prompt": prompt,
            "stream": False
        }
        
        ollama_req = urllib.request.Request(
            'http://localhost:11434/api/generate',
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(ollama_req) as response:
                result = json.loads(response.read().decode('utf-8'))
                risposta_llm = result.get("response", "").strip()
        except Exception as e:
            risposta_llm = f"[Errore di connessione a Ollama: {e}. Assicurati che Ollama sia in esecuzione sul tuo Mac!]"

        return {
            "query": req.query,
            "context": contesto_trovato,
            "context_score": round(vincitore.score, 4),
            "response": risposta_llm
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
