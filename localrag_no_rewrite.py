#
# Test local RAG with embedding Model from your choice
# Then, loop on all the models available on ollama to compare answers
#

import logging
import torch
import time
import ollama
from prettytable import PrettyTable
import os
from openai import OpenAI
import argparse

class CustomFormatter(logging.Formatter):

    grey     = "\x1b[38;20m"
    yellow   = "\x1b[33;20m"
    red      = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset    = "\x1b[0m"
    format   = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt   = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# create logger with 'spam_application'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

logger.info("START")

logger.debug('Setting Vars')
my_filename         = 'vault.txt'                       ### FILENAME
OLLAMA_HOST         = '192.168.0.1'                     ### OLLAMA HOST
text_to_remove      = " - Dernière modification le 19 mai 2024 - Document généré le 03 juin 2024"
EMBED_MODEL         = None
DEFAULT_EMBED_MODEL = 'llama3:8b-instruct-q8_0'
oclient             = ollama.Client(host=OLLAMA_HOST)
embed_models_list   = []
tchat_models_list   = []
models_list         = oclient.list()
# ANSI escape codes for colors
PINK                = '\033[95m'
CYAN                = '\033[96m'
YELLOW              = '\033[93m'
NEON_GREEN          = '\033[92m'
RESET_COLOR         = '\033[0m'

def look_for_embed_models():
    logger.debug('Finding available embed models')
    for model in models_list['models']:
        current_model_name = model['name']
        if "embed" not in current_model_name:
            continue
        else:
            embed_models_list.append(current_model_name)

def look_for_tchat_models():
    logger.debug('Finding available tchat models')
    for model in models_list['models']:
        current_model_name = model['name']
        if "embed" in current_model_name:
            continue
        else:
            tchat_models_list.append(current_model_name)

look_for_embed_models()
logger.debug('Creating PrettyTable')
table = PrettyTable()
table.field_names = ["ID", "Name"]
# Ajouter les données au tableau
i = 1
for item in embed_models_list:
    table.add_row([i, item])
    i += 1

logger.debug('Display PrettyTable')
print(table)

while EMBED_MODEL is None:
    try:
        choix = int(input("Veuillez choisir un modèle pour l'embedding en entrant l'ID correspondant : "))
        # Vérifier si le choix est valide et paramétrer le workspace
        choix = choix - 1
        EMBED_MODEL = embed_models_list[choix]
        print(f"Vous avez choisi le model pour embed : {EMBED_MODEL}")
    except ValueError:
        print("Entrée invalide. Veuillez entrer un nombre correspondant à l'ID.")

logger.info(f"OLLAMA HOST : {OLLAMA_HOST}")
logger.info(f"EMBED MODEL : {EMBED_MODEL}")
input("Press Enter to continue... or CTRL+C to abort")

logger.debug("Setting Vars : Done.")

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    logger.info(f"User input : {rewritten_input}")
    #logger.info(vault_embeddings)
    #logger.info(vault_content)
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = oclient.embeddings(model=EMBED_MODEL, prompt=rewritten_input)["embedding"]
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

# Function to interact with the Ollama model
def ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, ollama_model, conversation_history):
    # Get relevant context from the vault
    relevant_context = get_relevant_context(user_input, vault_embeddings_tensor, vault_content, top_k=3)
    if relevant_context:
        # Convert list to a single string with newlines between items
        context_str = "\n".join(relevant_context)
        logger.info("Context Pulled from Documents :")
        print(f"{CYAN} {context_str} {RESET_COLOR}")
    else:
        print(f"{CYAN}No relevant context found.{RESET_COLOR}")
    
    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input
    
    # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input_with_context})
    
    # Create a message history including the system message and the conversation history
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    # Send the completion request to the Ollama model
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages
    )
    
    # Append the model's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    # Return the content of the response from the model
    return response.choices[0].message.content

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default=DEFAULT_EMBED_MODEL, help="Ollama model to use (default: llama3)")
args = parser.parse_args()

# Configuration for the Ollama API client
client = OpenAI(
    base_url = f'http://{OLLAMA_HOST}:11434/v1',
    api_key='llama3'
)

# Load the vault content
vault_content = []
if os.path.exists(my_filename):
    with open(my_filename, "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()

# Generate embeddings for the vault content using Ollama
logger.debug("Generate Embeddings")
print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
vault_embeddings = []

tic = time.perf_counter()
for content in vault_content:
    print(YELLOW + 'QUERY' + RESET_COLOR)
    if text_to_remove:
        content = content.replace(text_to_remove, ' ')
        print(content)
    response = oclient.embeddings(model=EMBED_MODEL, prompt=content)
    
    print(NEON_GREEN + 'Done' + RESET_COLOR)
    print(YELLOW + 'APPEND' + RESET_COLOR)
    vault_embeddings.append(response["embedding"])
    print(NEON_GREEN + 'Done' + RESET_COLOR)

logger.debug("Generate Embeddings : Done.")
toc = time.perf_counter()

if (toc - tic) >= 1000:
    duration = toc - tic / 60
    print(f"Duration : {duration:0.4f} minutes")
else:
    print(f"Duration : {toc - tic:0.4f} seconds")


# Convert to tensor and print embeddings
vault_embeddings_tensor = torch.tensor(vault_embeddings) 
print("Embeddings for each line in the vault:")
print(vault_embeddings_tensor)

# Conversation loop
conversation_history = []
system_message = "Vous êtes un assistant utile et expert dans l'extraction des informations les plus utiles d'un texte donné. Apportez également des informations supplémentaires pertinentes à la requête de l'utilisateur en dehors du contexte donné."

user_query = None
look_for_tchat_models()

for model in tchat_models_list:
    logger.info(f"== Tchat Model : {model}")
    while True:
        try:
            user_query = input(YELLOW + "Ask a question about your documents (or type 'quit' to switch to next Model / CTRL+C to exit loop): " + RESET_COLOR)
        except KeyboardInterrupt:
            raise SystemExit
        if user_query.lower() == 'quit':
            break
        response = ollama_chat(user_query, system_message, vault_embeddings_tensor, vault_content, model, conversation_history)
        #response = ollama_chat(user_query, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
        print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)

logger.info("End of File")