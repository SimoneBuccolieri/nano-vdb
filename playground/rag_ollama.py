import json
import urllib.request
from sentence_transformers import SentenceTransformer
import nano_vdb

print("=== FASE 3: L'APP RAG COMPLETA ===")

# --- PARTE 1: RETRIEVAL (Recupero Semantico) ---
print("\n1. Caricamento Embedding Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
DIMS = model.get_sentence_embedding_dimension()
db = nano_vdb.NanoVectorDB(DIMS)

documenti = [
    "Roma è la capitale dell'Italia e ospita il famoso Colosseo.",
    "La pizza Margherita è un piatto tipico nato a Napoli nel 1889.",
    "Il Giappone è una nazione insulare situata nell'Oceano Pacifico.",
    "L'intelligenza artificiale generativa e gli LLM stanno rivoluzionando la tecnologia.",
    "Il Monte Bianco è la montagna più alta della catena delle Alpi.",
    "Steve Jobs e Steve Wozniak hanno fondato la Apple nel garage dei loro genitori."
]

print("   Popolamento DB C++...")
embeddings = model.encode(documenti)
for i, (testo, vettore) in enumerate(zip(documenti, embeddings)):
    db.insert(f"doc_{i}", vettore.tolist())

domanda = "Chi ha inventato la playstation?"
print(f"\nDomanda dell'Utente: '{domanda}'")

# Il nostro DB C++ trova l'informazione giusta
vettore_domanda = model.encode(domanda).tolist()
risultati = db.search_graph(vettore_domanda, top_k=1)
vincitore = risultati[0]
indice_vincitore = int(vincitore.id.split('_')[1])
contesto_trovato = documenti[indice_vincitore]

print(f"-> Contesto recuperato dal DB (Score: {vincitore.score:.2f}): '{contesto_trovato}'")

# --- PARTE 2: GENERATION (Generazione con LLM via Ollama) ---
print("\n2. Passaggio delle informazioni all'LLM (Ollama - phi3:mini)...")

# Creiamo il classico "Prompt RAG"
prompt = f"""Sei un assistente intelligente. Usa SOLO le informazioni fornite nel seguente contesto per rispondere alla domanda dell'utente. Se l'informazione non è nel contesto, rispondi 'Non lo so'.

CONTESTO:
{contesto_trovato}

DOMANDA DELL'UTENTE:
{domanda}

RISPOSTA BREVE IN ITALIANO:
"""

data = {
    "model": "phi3:mini",
    "prompt": prompt,
    "stream": False # Vogliamo tutta la risposta intera, non parola per parola
}

req = urllib.request.Request(
    'http://localhost:11434/api/generate',
    data=json.dumps(data).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        risposta_llm = result.get("response", "")
        print("\n=== RISPOSTA FINALE GENERATA DALLA MACCHINA ===")
        print(risposta_llm.strip())
except Exception as e:
    print(f"\nErrore di connessione a Ollama: {e}")
    print("Assicurati che Ollama sia in esecuzione in background sul tuo Mac!")
