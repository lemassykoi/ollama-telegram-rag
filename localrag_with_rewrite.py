import logging
import time
import torch
import ollama
from prettytable import PrettyTable
import os
import json

# Configurer le logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\t(%(filename)s:%(lineno)d)',
    level=logging.INFO
)

logger.info("START")

logger.debug("Setting Vars")
my_filename       = "vault.txt"                    ### FILENAME
OLLAMA_HOST       = '127.0.0.1'                  ### OLLAMA HOST
text_to_remove    = ""
oclient           = ollama.Client(host=OLLAMA_HOST)
EMBED_MODEL       = None
CHAT_MODEL        = "llama3-custom"
embed_models_list = []
models_list       = oclient.list()
# ANSI escape codes for console colors
PINK              = '\033[95m'
CYAN              = '\033[96m'
YELLOW            = '\033[93m'
NEON_GREEN        = '\033[92m'
RESET_COLOR       = '\033[0m'

logger.debug("Finding available models")
for model in models_list['models']:
    current_model_name = model['name']
    if "embed" not in current_model_name:
        continue
    else:
        embed_models_list.append(current_model_name)

logger.debug("Creating PrettyTable")
table = PrettyTable()
table.field_names = ["ID", "Name"]
# Ajouter les données au tableau
i = 1
for item in embed_models_list:
    table.add_row([i, item])
    i += 1

logger.debug("Display PrettyTable")
print(table)

while EMBED_MODEL is None:
    try:
        choix = int(input("Veuillez choisir un modèle pour l'embedding en entrant l'ID correspondant : "))
        choix = choix - 1
        EMBED_MODEL = embed_models_list[choix]
        print(f"Vous avez choisi le model pour embed : {EMBED_MODEL}")
    except ValueError:
        print("Entrée invalide. Veuillez entrer un nombre correspondant à l'ID.")

logger.info(NEON_GREEN + f"OLLAMA HOST : {OLLAMA_HOST}" + RESET_COLOR)
logger.info(NEON_GREEN + f"EMBED MODEL : {EMBED_MODEL}" + RESET_COLOR)
logger.info(NEON_GREEN + f"CHAT  MODEL : {CHAT_MODEL}" + RESET_COLOR)
logger.info(NEON_GREEN + 'Is Vault Present?\t: ' + YELLOW + str(os.path.isfile(my_filename)) + RESET_COLOR)

input("\nPress Enter to continue... or CTRL+C to abort")

logger.debug("Setting Vars : Done.")

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
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

def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Réécrivez la requête suivante en incorporant le contexte pertinent de l'historique de la conversation.
    La requête réécrite doit :

    - Préserver l'intention fondamentale et la signification de la requête d'origine
    - Développer et clarifier la requête pour la rendre plus spécifique et informative pour récupérer le contexte pertinent
    - Évitez d'introduire de nouveaux sujets ou requêtes qui s'écartent de la requête d'origine
    - NE JAMAIS RÉPONDRE à la requête d'origine, mais concentrez-vous plutôt sur sa reformulation et son extension dans une nouvelle requête

    Renvoie UNIQUEMENT le texte de la requête réécrit, sans aucune mise en forme ni explication supplémentaire.
    
    Historique de la conversation:
    {context}
    
    Requête originale: [{user_input}]
    
    Requête réécrite: 
    """
    message = [{"role": "system", "content": prompt}]

    ## Ollama
    logger.info(YELLOW + 'Rewrite Query' + RESET_COLOR)
    full_response = oclient.chat(model=ollama_model, messages=message)  ## This full response contains duration
    rewritten_query = full_response.get('message').get('content')
    logger.info(PINK + f'Rewriten Query : {rewritten_query}' + RESET_COLOR)
    return json.dumps({"Rewritten Query": rewritten_query})
   
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})
    
    if len(conversation_history) > 1:
        query_json = {
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        logger.info(PINK + "Original Query: " + user_input + RESET_COLOR)
        logger.info(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)
    else:
        rewritten_query = user_input
    
    relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        logger.info("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        logger.info(CYAN + "No relevant context found." + RESET_COLOR)
    
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
    
    conversation_history[-1]["content"] = user_input_with_context
    
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    logger.info(YELLOW + 'Ollama Chat' + RESET_COLOR)
    full_response = oclient.chat(model=ollama_model, messages=messages)  ## This full response contains duration
    answer = full_response.get('message').get('content')
    conversation_history.append({"role": "assistant", "content": answer})
    return answer

# Load the vault content
logger.info(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()

# Generate embeddings for the vault content using Ollama
logger.info(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
vault_embeddings = []

tic = time.perf_counter()
for content in vault_content:
    logger.info(YELLOW + 'QUERY' + RESET_COLOR)
    logger.info(content)
    if text_to_remove:
        content = content.replace(text_to_remove, ' ')
    response = oclient.embeddings(model=EMBED_MODEL, prompt=content)
    
    logger.info(NEON_GREEN + 'Done' + RESET_COLOR)
    logger.info(YELLOW + 'APPEND' + RESET_COLOR)
    vault_embeddings.append(response["embedding"])
    logger.info(NEON_GREEN + 'Done' + RESET_COLOR)

logger.debug("Generate Embeddings : Done.")
toc = time.perf_counter()

if (toc - tic) >= 1000:
    duration = toc - tic / 60
    logger.info(f"Duration : {duration:0.4f} minutes")
else:
    logger.info(f"Duration : {toc - tic:0.4f} seconds")

# Convert to tensor and print embeddings
logger.info("Converting embeddings to tensor...")
vault_embeddings_tensor = torch.tensor(vault_embeddings) 
logger.info("Embeddings for each line in the vault:")
logger.info(vault_embeddings_tensor)

# Conversation loop
logger.info("Starting conversation loop...")
conversation_history = []
system_message = "Vous êtes un assistant utile et expert dans l'extraction des informations les plus utiles d'un texte donné. Apportez également des informations supplémentaires pertinentes à la requête de l'utilisateur en dehors du contexte donné."

while True:
    logger.debug("User Loop")
    user_input = input(YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + RESET_COLOR)
    if user_input.lower() == 'quit':
        break
    
    response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, CHAT_MODEL, conversation_history)
    logger.info(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)

logger.info("End of File")