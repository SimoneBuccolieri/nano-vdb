from sentence_transformers import SentenceTransformer
import nano_vdb

print("=== FASE 2: EMBEDDINGS VERI ===")

# 1. Carichiamo il modello AI locale. 
# Scaricherà automaticamente i "pesi" dell'intelligenza artificiale (circa 80MB).
# Questo modello capisce il significato delle frasi (anche in italiano!) e le trasforma in vettori.
print("\nCaricamento del modello AI (potrebbe volerci qualche secondo)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Scopriamo quante dimensioni ha il vettore prodotto da questo specifico modello
DIMS = model.get_sentence_embedding_dimension()
print(f"Il modello produce vettori a {DIMS} dimensioni.")

# 2. Inizializziamo il nostro database C++ con le dimensioni esatte del modello!
db = nano_vdb.NanoVectorDB(DIMS)

# 3. Creiamo la nostra base di conoscenza (il database testuale)
documenti = [
    "Roma è la capitale dell'Italia e ospita il famoso Colosseo.",
    "La pizza Margherita è un piatto tipico nato a Napoli nel 1889.",
    "Il Giappone è una nazione insulare situata nell'Oceano Pacifico.",
    "L'intelligenza artificiale generativa e gli LLM stanno rivoluzionando la tecnologia.",
    "Il Monte Bianco è la montagna più alta della catena delle Alpi.",
    "Steve Jobs e Steve Wozniak hanno fondato la Apple nel garage dei loro genitori."
]

print("\nTraduzione del testo in vettori (Embeddings) e salvataggio nel DB C++...")
# Il modello AI "legge" le frasi e le traduce in coordinate matematiche
embeddings = model.encode(documenti)

# Salviamo ogni coordinata matematica nel nostro database per costruire il grafo
for i, (testo, vettore) in enumerate(zip(documenti, embeddings)):
    db.insert(f"doc_{i}", vettore.tolist())
    
print("Inserimento e costruzione del grafo HNSW completati!")

# 4. IL TEST DELLA VERITÀ: La Ricerca Semantica
# Facciamo una domanda che NON contiene le stesse identiche parole del testo nel DB
domanda = "Qual è la capitale?"
print(f"\nDomanda dell'utente: '{domanda}'")

# Traduciamo la domanda in coordinate usando lo stesso modello
vettore_domanda = model.encode(domanda).tolist()

# Usiamo il nostro DB per trovare la coordinata salvata più vicina
risultati = db.search_graph(vettore_domanda, top_k=1)

vincitore = risultati[0]
# Recuperiamo l'indice dal nostro ID fittizio (es. da "doc_4" tiriamo fuori il 4)
indice_vincitore = int(vincitore.id.split('_')[1])
testo_trovato = documenti[indice_vincitore]

print("\n=== RISULTATO ===")
print(f"Score di Somiglianza: {vincitore.score:.4f} (1.0 = identico)")
print(f"Testo Recuperato dal DB: '{testo_trovato}'")
