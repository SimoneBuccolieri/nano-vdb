import os
import urllib.request
import sys

MODEL_URL = "https://huggingface.co/lmstudio-community/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
MODEL_PATH = "models/gemma-2-2b-it-Q4_K_M.gguf"

def progress_bar(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    percent = min(100, percent)
    downloaded = count * block_size / (1024 * 1024)
    total = total_size / (1024 * 1024)
    sys.stdout.write(f"\rScaricamento: {percent}% | {downloaded:.1f}/{total:.1f} MB")
    sys.stdout.flush()

def main():
    os.makedirs("models", exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print(f"Il file {MODEL_PATH} esiste già. Salto il download.")
        return

    print(f"Inizio download di {MODEL_PATH}...")
    print(f"Sorgente: {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, progress_bar)
        print("\nDownload completato con successo!")
    except Exception as e:
        print(f"\nErrore durante il download: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        sys.exit(1)

if __name__ == "__main__":
    main()
