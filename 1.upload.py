import logging
import tkinter as tk
from tkinter import filedialog
import PyPDF2
import re
import json
import time

my_filename = "vault.txt"

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger.info("START")

logger.info(f"Vault File Name : {my_filename}")

logger.debug("Setting Functions")
""" # Function to start the progress bar animation
def start_progress_bar():
    progress_bar.pack(pady=10)
    progress_bar.start()

# Function to stop the progress bar animation
def stop_progress_bar():
    progress_bar.stop()
    progress_bar.pack_forget() """

# Function to make the label text blink
def blink_text():
    current_color = processing_label.cget("foreground")
    next_color = "red" if current_color == "black" else "black"
    processing_label.config(foreground=next_color)
    root.after(500, blink_text)

# Function to show the processing label
def show_processing_label():
    processing_label.pack(pady=10)
    root.update_idletasks()
    blink_text()

# Function to hide the processing label
def hide_processing_label():
    processing_label.pack_forget()

# Function to convert PDF to text and append to vault.txt
def convert_pdf_to_text():
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        show_processing_label()
        root.update_idletasks()
        logger.info(f'Received File : {file_path}')
        logger.info('Please wait...')
        tic = time.perf_counter()
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            text = ''
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                if page.extract_text():
                    text += page.extract_text() + " "
            
            # Normalize whitespace and clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split text into chunks by sentences, respecting a maximum chunk size
            sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                # Check if the current sentence plus the current chunk exceeds the limit
                if len(current_chunk) + len(sentence) + 1 < 2048:  # +1 for the space
                    current_chunk += (sentence + " ").strip()
                else:
                    # When the chunk exceeds 1024 characters, store it and start a new one
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:  # Don't forget the last chunk!
                chunks.append(current_chunk)
            with open(my_filename, "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    # Write each chunk to its own line
                    vault_file.write(chunk.strip() + u"\n")  # Two newlines to separate chunks
                    #vault_file.write(chunk.strip() + '\\\\n')  # Two newlines to separate chunks
            logger.info(f"PDF content appended to {my_filename} with each chunk on a separate line.")
        toc = time.perf_counter()
        logger.info(f"Duration : {toc - tic:0.4f} seconds")
        hide_processing_label()
        enable_start_button()

# Function to upload a text file and append to vault.txt
def upload_txtfile():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        show_processing_label()
        root.update_idletasks()
        logger.info(f'Received File : {file_path}')
        logger.info('Please wait...')
        tic = time.perf_counter()
        with open(file_path, 'r', encoding="utf-8") as txt_file:
            text = txt_file.read()
            
            # Normalize whitespace and clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split text into chunks by sentences, respecting a maximum chunk size
            sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                # Check if the current sentence plus the current chunk exceeds the limit
                if len(current_chunk) + len(sentence) + 1 < 1024:  # +1 for the space
                    current_chunk += (sentence + " ").strip()
                else:
                    # When the chunk exceeds 1024 characters, store it and start a new one
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:  # Don't forget the last chunk!
                chunks.append(current_chunk)
            with open(my_filename, "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    # Write each chunk to its own line
                    vault_file.write(chunk.strip() + u"\n")  # Two newlines to separate chunks
            logger.info(f"Text file content appended to {my_filename} with each chunk on a separate line.")
        toc = time.perf_counter()
        logger.info(f"Duration : {toc - tic:0.4f} seconds")
        hide_processing_label()
        enable_start_button()

# Function to upload a JSON file and append to vault.txt
def upload_jsonfile():
    file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if file_path:
        show_processing_label()
        root.update_idletasks()
        logger.info(f'Received File : {file_path}')
        logger.info('Please wait...')
        tic = time.perf_counter()
        with open(file_path, 'r', encoding="utf-8") as json_file:
            data = json.load(json_file)
            
            # Flatten the JSON data into a single string
            text = json.dumps(data, ensure_ascii=False)
            
            # Normalize whitespace and clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split text into chunks by sentences, respecting a maximum chunk size
            sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                # Check if the current sentence plus the current chunk exceeds the limit
                if len(current_chunk) + len(sentence) + 1 < 1024:  # +1 for the space
                    current_chunk += (sentence + " ").strip()
                else:
                    # When the chunk exceeds 1024 characters, store it and start a new one
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:  # Don't forget the last chunk!
                chunks.append(current_chunk)
            with open(my_filename, "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    # Write each chunk to its own line
                    vault_file.write(chunk.strip() + u"\n")  # Two newlines to separate chunks
            logger.info(f"JSON file content appended to {my_filename} with each chunk on a separate line.")
        toc = time.perf_counter()
        logger.info(f"Duration : {toc - tic:0.4f} seconds")
        hide_processing_label()
        enable_start_button()

def enable_start_button():
    local_rag_button.config(state=tk.NORMAL)
    local_rag_no_rewrite_button.config(state=tk.NORMAL)

def start_local_rag_with_rewrite():
    logger.info('Starting Local RAG...')
    import subprocess
    try:
        subprocess.run(["python", "localrag_with_rewrite.py"])
    except KeyboardInterrupt:
        raise SystemExit

def start_local_rag_no_rewrite():
    logger.info('Starting Local RAG with No Rewrite...')
    import subprocess
    try:
        subprocess.run(["python", "localrag_no_rewrite.py"])
    except KeyboardInterrupt:
        raise SystemExit 

logger.debug("Setting Functions : Done.")

logger.debug("Create the main window")
root = tk.Tk()
root.title("Upload .pdf, .txt, or .json")
root.geometry("300x300")

logger.debug("Create a button to open the file dialog for PDF")
pdf_button = tk.Button(root, text="Upload PDF", command=convert_pdf_to_text)
pdf_button.pack(pady=10)

logger.debug("Create a button to open the file dialog for TXT")
txt_button = tk.Button(root, text="Upload Text File", command=upload_txtfile)
txt_button.pack(pady=10)

logger.debug("Create a button to open the file dialog for JSON")
json_button = tk.Button(root, text="Upload JSON File", command=upload_jsonfile)
json_button.pack(pady=10)

logger.debug("Create a button to Start Local RAG")
local_rag_button = tk.Button(root, text="Start Local RAG", command=start_local_rag_with_rewrite, state=tk.DISABLED)
local_rag_button.pack(pady=10)

logger.debug("Create a button to Start Local RAG with NO Rewrite")
local_rag_no_rewrite_button = tk.Button(root, text="Start Local RAG\n(No Rewrite)", command=start_local_rag_no_rewrite, state=tk.DISABLED)
local_rag_no_rewrite_button.pack(pady=10)

logger.debug("Create a progress bar")
processing_label = tk.Label(root, text="Processing...", foreground="black")

logger.debug("Run the main event loop")
root.mainloop()