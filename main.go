package main

import (
	"fmt"
	"io"
	"net/http"
	"time"
)

// PORTA su cui gira il server Go
const GoPort = ":8080"

// INDIRIZZO del microservizio Python FastAPI
const PythonServiceURL = "http://127.0.0.1:8000"

// Handler che funge da Proxy verso il microservizio Python
func proxyHandler(w http.ResponseWriter, r *http.Request) {
	// Determiniamo il path della rotta (es. /api/search -> /search)
	endpoint := r.URL.Path[4:] // Rimuove "/api"
	targetURL := fmt.Sprintf("%s%s", PythonServiceURL, endpoint)

	// Creiamo una nuova richiesta HTTP diretta a Python
	proxyReq, err := http.NewRequest(r.Method, targetURL, r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("Errore creazione richiesta proxy: %v", err), http.StatusInternalServerError)
		return
	}

	// Copiamo gli header originali della richiesta
	for name, values := range r.Header {
		for _, value := range values {
			proxyReq.Header.Add(name, value)
		}
	}

	// Eseguiamo la richiesta verso il microservizio Python
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(proxyReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("Errore connessione a Python FastAPI: %v. Assicurati che server.py sia in esecuzione!", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Copiamo gli header della risposta da Python a Go
	for name, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(name, value)
		}
	}
	w.WriteHeader(resp.StatusCode)

	// Copiamo il corpo della risposta da Python a Go
	_, err = io.Copy(w, resp.Body)
	if err != nil {
		fmt.Printf("Errore copia risposta body: %v\n", err)
	}
}

func main() {
	// Definiamo i file statici del frontend nella cartella "static"
	fileServer := http.FileServer(http.Dir("./static"))
	http.Handle("/", fileServer)

	// Definiamo le rotte proxy verso Python
	http.HandleFunc("/api/status", proxyHandler)
	http.HandleFunc("/api/insert", proxyHandler)
	http.HandleFunc("/api/search", proxyHandler)
	http.HandleFunc("/api/rag", proxyHandler)

	fmt.Printf("=== SERVER GO WEB IN ESECUZIONE ===\n")
	fmt.Printf("Apri il tuo browser all'indirizzo: http://localhost%s\n", GoPort)
	fmt.Printf("Assicurati che anche uvicorn (FastAPI) sia attivo sulla porta 8000.\n\n")

	err := http.ListenAndServe(GoPort, nil)
	if err != nil {
		fmt.Printf("Errore avvio server Go: %v\n", err)
	}
}
