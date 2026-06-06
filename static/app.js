document.addEventListener("DOMContentLoaded", () => {
    // Riferimenti agli elementi DOM
    const dbDimensionsEl = document.getElementById("db-dimensions");
    const dbSizeEl = document.getElementById("db-size");
    const dbDocListEl = document.getElementById("db-doc-list");

    const insertForm = document.getElementById("insert-form");
    const docIdInput = document.getElementById("doc-id");
    const docTextInput = document.getElementById("doc-text");
    const insertMessageEl = document.getElementById("insert-message");

    const searchQueryInput = document.getElementById("search-query");
    const searchTopKSelect = document.getElementById("search-top-k");
    const searchModeSelect = document.getElementById("search-mode");
    const btnSearch = document.getElementById("btn-search");
    const searchResultsEl = document.getElementById("search-results");

    const ragQueryInput = document.getElementById("rag-query");
    const btnRag = document.getElementById("btn-rag");
    const ragContextEl = document.getElementById("rag-context");
    const ragResponseEl = document.getElementById("rag-response");

    // Funzione helper per mostrare/nascondere il caricamento
    function setLoading(element, isLoading, placeholderText = "Elaborazione in corso...") {
        if (isLoading) {
            element.innerHTML = `<p class="placeholder-text"><span class="loading-spinner"></span> ${placeholderText}</p>`;
        }
    }

    // 1. CARICAMENTO STATO DATABASE
    async function fetchDbStatus() {
        try {
            const response = await fetch("/api/status");
            if (!response.ok) throw new Error("Errore durante il recupero dello stato.");
            
            const data = await response.json();
            dbDimensionsEl.textContent = data.dimensions;
            dbSizeEl.textContent = data.size;

            // Popolamento lista documenti
            dbDocListEl.innerHTML = "";
            if (data.documents.length === 0) {
                dbDocListEl.innerHTML = "<li>Nessun documento indicizzato</li>";
            } else {
                data.documents.forEach((doc, idx) => {
                    const li = document.createElement("li");
                    li.textContent = doc;
                    li.title = doc; // Mostra tooltip al passaggio del mouse
                    dbDocListEl.appendChild(li);
                });
            }
        } catch (error) {
            console.error("Errore recupero stato:", error);
            dbDimensionsEl.textContent = "Errore";
            dbSizeEl.textContent = "Errore";
        }
    }

    // 2. INSERIMENTO DOCUMENTO
    insertForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const docId = docIdInput.value.trim();
        const docText = docTextInput.value.trim();
        
        insertMessageEl.className = "message";
        insertMessageEl.textContent = "Quantizzazione in corso...";

        try {
            const response = await fetch("/api/insert", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ id: docId, text: docText })
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Errore sconosciuto.");

            insertMessageEl.className = "message success";
            insertMessageEl.textContent = `Documento ${data.id} inserito e quantizzato con successo!`;
            
            // Resettiamo il form e aggiorniamo il database status
            docIdInput.value = "";
            docTextInput.value = "";
            fetchDbStatus();

            setTimeout(() => {
                insertMessageEl.textContent = "";
                insertMessageEl.className = "message";
            }, 4000);

        } catch (error) {
            insertMessageEl.className = "message error";
            insertMessageEl.textContent = `Errore: ${error.message}`;
        }
    });

    // 3. RICERCA SEMANTICA
    btnSearch.addEventListener("click", async () => {
        const query = searchQueryInput.value.trim();
        if (!query) return;

        const topK = parseInt(searchTopKSelect.value);
        const mode = searchModeSelect.value;

        setLoading(searchResultsEl, true, "Esecuzione ricerca semantica sul database quantizzato...");

        const startTime = performance.now();

        try {
            const response = await fetch("/api/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query, top_k: topK, mode: mode })
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Errore sconosciuto.");

            const endTime = performance.now();
            const elapsedMs = (endTime - startTime).toFixed(1);

            searchResultsEl.innerHTML = "";
            if (data.results.length === 0) {
                searchResultsEl.innerHTML = `<p class="placeholder-text">Nessun risultato trovato (${elapsedMs} ms).</p>`;
                return;
            }

            // Mostriamo il tempo impiegato
            const infoText = document.createElement("p");
            infoText.className = "placeholder-text";
            infoText.style.marginBottom = "0.75rem";
            infoText.textContent = `Trovati ${data.results.length} risultati in ${elapsedMs} ms (metodo: ${mode === 'search' ? 'Grafo HNSW' : 'Scansione Lineare'}).`;
            searchResultsEl.appendChild(infoText);

            // Generiamo i blocchi dei risultati
            data.results.forEach(res => {
                const item = document.createElement("div");
                item.className = "result-item";
                item.innerHTML = `
                    <div class="result-header">
                        <span class="result-id">${res.id}</span>
                        <span class="result-score">Score: ${res.score}</span>
                    </div>
                    <p class="result-text">${res.text}</p>
                `;
                searchResultsEl.appendChild(item);
            });

        } catch (error) {
            searchResultsEl.innerHTML = `<p class="message error">Errore: ${error.message}</p>`;
        }
    });

    // 4. CHIAMATA ASSISTENTE RAG (OLLAMA)
    btnRag.addEventListener("click", async () => {
        const query = ragQueryInput.value.trim();
        if (!query) return;

        setLoading(ragContextEl, true, "Cerca contesto nel database C++...");
        setLoading(ragResponseEl, true, "In attesa della risposta di Ollama (phi3:mini)...");

        try {
            const response = await fetch("/api/rag", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query })
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Errore sconosciuto.");

            // Mostra il contesto
            ragContextEl.innerHTML = `
                <p><strong>[${data.context_score ? 'Score: ' + data.context_score : ''}]</strong> ${data.context || 'Nessun contesto trovato.'}</p>
            `;

            // Mostra la risposta generata
            ragResponseEl.innerHTML = `<p>${data.response}</p>`;

        } catch (error) {
            ragContextEl.innerHTML = `<p class="message error">Errore: ${error.message}</p>`;
            ragResponseEl.innerHTML = `<p class="placeholder-text">Operazione interrotta.</p>`;
        }
    });

    // Carichiamo lo stato iniziale
    fetchDbStatus();
});
