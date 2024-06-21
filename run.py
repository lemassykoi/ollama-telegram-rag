import logging
import os
import aiohttp
import json
import torch
import ollama
from openai import OpenAI
from aiogram import types
from aiohttp import ClientTimeout
from asyncio import Lock
from functools import wraps
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.filters.command import Command, CommandStart
from aiogram.types import Message
from aiogram.utils.keyboard import InlineKeyboardBuilder
import asyncio
import traceback
import io
import time
import base64

## interactions
class contextLock:
    lock = Lock()

    async def __aenter__(self):
        await self.lock.acquire()

    async def __aexit__(self, exc_type, exc_value, exc_traceback):
        self.lock.release()

# Get content of .env file
load_dotenv()
token                   = os.getenv("TOKEN")
ollama_base_url         = os.getenv("OLLAMA_BASE_URL", "127.0.0.1")
ollama_port             = os.getenv("OLLAMA_PORT", "11434")
log_level_str           = os.getenv("LOG_LEVEL", "INFO")
modelname               = os.getenv("INITMODEL", "llama3")
log_levels              = list(logging._levelToName.values())
allowed_ids             = list(map(int, os.getenv("USER_IDS", "").split(",")))
admin_ids               = list(map(int, os.getenv("ADMIN_IDS", "").split(",")))
timeout                 = os.getenv("TIMEOUT", "3000")
log_format              = "%(asctime)s - %(name)s - %(levelname)s - %(message)s \t\t\t\t (%(filename)s:%(lineno)d)"
#EMBED_MODEL             = 'mxbai-embed-large'
EMBED_MODEL             = 'snowflake-arctic-embed'
oclient                 = ollama.Client(host=ollama_base_url)
ACTIVE_CHATS            = {}
ACTIVE_CHATS_LOCK       = contextLock()
mention                 = None
CHAT_TYPE_GROUP         = "group"
CHAT_TYPE_SUPERGROUP    = "supergroup"
my_filename             = 'vault.txt'
conversation_history    = []
system_message          = "Vous êtes un assistant utile et expert dans l'extraction des informations les plus utiles d'un texte donné. Apportez également des informations supplémentaires pertinentes à la requête de l'utilisateur en dehors du contexte donné."
vault_embeddings_tensor = None

# Configuration for the Ollama API client
client = OpenAI(
    base_url = f'http://{ollama_base_url}:11434/v1',
    api_key='llama3'
)

if log_level_str not in log_levels:
    log_level = logging.DEBUG
else:
    log_level = logging.getLevelName(log_level_str)

logging.basicConfig(
    format=log_format,
    level='INFO'  ## force info / should be : log_level
)
current_log_level = logging.getLevelName(logging.root.level)
if current_log_level != log_level_str:
    logging.warning('CAUTION : LOG LEVEL IS DIFFERENT FROM OS ENV')

logging = logging.getLogger(__name__)
logging.info(f"Log Level from OS : {log_level_str}")
logging.info(f"Current Log Level : {current_log_level}")

async def model_list():
    async with aiohttp.ClientSession() as session:
        url = f"http://{ollama_base_url}:{ollama_port}/api/tags"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data["models"]
            else:
                return []
async def generate(payload: dict, modelname: str, prompt: str):
    client_timeout = ClientTimeout(total=int(timeout))
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        url = f"http://{ollama_base_url}:{ollama_port}/api/chat"

        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise aiohttp.ClientResponseError(
                        status=response.status, message=response.reason
                    )
                buffer = b""

                async for chunk in response.content.iter_any():
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if line:
                            yield json.loads(line)
        except aiohttp.ClientError as e:
            print(f"Error during request: {e}")
def perms_allowed(func):
    @wraps(func)
    async def wrapper(message: types.Message = None, query: types.CallbackQuery = None):
        user_id = message.from_user.id if message else query.from_user.id
        if user_id in admin_ids or user_id in allowed_ids:
            if message:
                return await func(message)
            elif query:
                return await func(query=query)
        else:
            if message:
                if message and message.chat.type in ["supergroup", "group"]:
                    return
                await message.answer("Access Denied")
            elif query:
                if message and message.chat.type in ["supergroup", "group"]:
                    return
                await query.answer("Access Denied")

    return wrapper
def perms_admins(func):
    @wraps(func)
    async def wrapper(message: types.Message = None, query: types.CallbackQuery = None):
        user_id = message.from_user.id if message else query.from_user.id
        if user_id in admin_ids:
            if message:
                return await func(message)
            elif query:
                return await func(query=query)
        else:
            if message:
                if message and message.chat.type in ["supergroup", "group"]:
                    return
                await message.answer("Access Denied")
                logging.info(
                    f"[MSG] {message.from_user.first_name} {message.from_user.last_name}({message.from_user.id}) is not allowed to use this bot."
                )
            elif query:
                if message and message.chat.type in ["supergroup", "group"]:
                    return
                await query.answer("Access Denied")
                logging.info(
                    f"[QUERY] {message.from_user.first_name} {message.from_user.last_name}({message.from_user.id}) is not allowed to use this bot."
                )

    return wrapper
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    logging.info('FUNC : Get Relevant Context')
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
    logging.info('FUNC : Get Relevant Context : DONE.')
    return relevant_context
## END of interactions

logging.info('Start...')
bot                  = Bot(token=token)
dp                   = Dispatcher()
start_kb             = InlineKeyboardBuilder()
settings_kb          = InlineKeyboardBuilder()
commands             = [
    types.BotCommand(command="start", description="Start"),
    types.BotCommand(command="reset", description="Reset Chat"),
    types.BotCommand(command="history", description="Look through messages"),
]

logging.info(f'Model Name : {modelname}')
start_kb.row(
    types.InlineKeyboardButton(text="ℹ️ About", callback_data="about"),
    types.InlineKeyboardButton(text="⚙️ Settings", callback_data="settings"),
)
settings_kb.row(
    types.InlineKeyboardButton(text="🔄 Switch LLM", callback_data="switchllm"),
    types.InlineKeyboardButton(text="✏️ Edit system prompt", callback_data="editsystemprompt"),
)

def load_vault(filename):
    # Load the vault content
    vault_content = []
    if os.path.exists(filename):
        with open(filename, "r", encoding='utf-8') as vault_file:
            vault_content = vault_file.readlines()
    return vault_content
def generate_embeddings(vault_content):
    vault_embeddings = []
    for content in vault_content:
        logging.debug('== QUERY')
        logging.debug(content)
        response = oclient.embeddings(model=EMBED_MODEL, prompt=content)
        logging.debug('=== APPEND')
        vault_embeddings.append(response["embedding"])
        logging.debug('=== Append Done')
    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    logging.info('Tensor Ok')
    return vault_embeddings_tensor

def is_mentioned_in_group_or_supergroup(message):
    return message.chat.type in [CHAT_TYPE_GROUP, CHAT_TYPE_SUPERGROUP] and (
        (message.text is not None and message.text.startswith(mention))
        or (message.caption is not None and message.caption.startswith(mention))
    )
async def get_bot_info():
    global mention
    if mention is None:
        get = await bot.get_me()
        mention = f"@{get.username}"
    return mention
@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    start_message = f"Welcome, <b>{message.from_user.full_name}</b>!"
    await message.answer(
        start_message,
        parse_mode=ParseMode.HTML,
        reply_markup=start_kb.as_markup(),
        disable_web_page_preview=True,
    )
@dp.message(Command("reset"))
async def command_reset_handler(message: Message) -> None:
    logging.info('RESET Asked')
    logging.debug('RESET Check if User is Allowed')
    if message.from_user.id in allowed_ids:
        logging.debug('RESET Yes')
        logging.debug('RESET Check if User is in ACTIVE_CHATS')
        if message.from_user.id in ACTIVE_CHATS:
            logging.debug('RESET Is ActiveChats ? YES')
            async with ACTIVE_CHATS_LOCK:
                ACTIVE_CHATS.pop(message.from_user.id)
            logging.info(f"Chat has been reset for {message.from_user.first_name}")
            await bot.send_message(
                chat_id=message.chat.id,
                text="Chat has been reset",
            )
        else:
            logging.debug('RESET Is ActiveChats ? NO')
    else:
        logging.debug('RESET No')

@dp.message(Command("history"))
async def command_get_context_handler(message: Message) -> None:
    if message.from_user.id in allowed_ids:
        if message.from_user.id in ACTIVE_CHATS:
            messages = ACTIVE_CHATS.get(message.chat.id)["messages"]
            context = ""
            for msg in messages:
                context += f"*{msg['role'].capitalize()}*: {msg['content']}\n"
            await bot.send_message(
                chat_id=message.chat.id,
                text=context,
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await bot.send_message(
                chat_id=message.chat.id,
                text="No chat history available for this user",
            )
@dp.callback_query(lambda query: query.data == "settings")
async def settings_callback_handler(query: types.CallbackQuery):
    await bot.send_message(
        chat_id=query.message.chat.id,
        text=f"Choose the right option.",
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
        reply_markup=settings_kb.as_markup()
    )
@dp.callback_query(lambda query: query.data == "switchllm")
async def switchllm_callback_handler(query: types.CallbackQuery):
    models = await model_list()
    switchllm_builder = InlineKeyboardBuilder()
    for model in models:
        modelname = model["name"]
        modelfamilies = ""
        if model["details"]["families"]:
            modelicon = {"llama": "🦙", "clip": "📷"}
            try:
                modelfamilies = "".join(
                    [modelicon[family] for family in model["details"]["families"]]
                )
            except KeyError as e:
                modelfamilies = f"✨"
        switchllm_builder.row(
            types.InlineKeyboardButton(
                text=f"{modelname} {modelfamilies}", callback_data=f"model_{modelname}"
            )
        )
    await query.message.edit_text(
        f"{len(models)} models available.\n🦙 = Regular\n🦙📷 = Multimodal", reply_markup=switchllm_builder.as_markup(),
    )
@dp.callback_query(lambda query: query.data.startswith("model_"))
async def model_callback_handler(query: types.CallbackQuery):
    global modelname
    global modelfamily
    modelname = query.data.split("model_")[1]
    await query.answer(f"Chosen model: {modelname}")
@dp.callback_query(lambda query: query.data == "about")
@perms_admins
async def about_callback_handler(query: types.CallbackQuery):
    dotenv_model = os.getenv("INITMODEL")
    global modelname
    await bot.send_message(
        chat_id=query.message.chat.id,
        text=f"<b>Your LLMs</b>\nCurrently using: <code>{modelname}</code>\nDefault in .env: <code>{dotenv_model}</code>\nThis project is under <a href='https://github.com/lemassykoi/ollama-telegram-rag/blob/main/LICENSE'>MIT License.</a>\n<a href='https://github.com/lemassykoi/ollama-telegram-rag'>Source Code</a>",
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )
@dp.message()
@perms_allowed
async def handle_message(message: types.Message):
    await get_bot_info()
    if message.chat.type == "private":
        await ollama_request(message)
    if is_mentioned_in_group_or_supergroup(message):
        if message.text is not None:
            text_without_mention = message.text.replace(mention, "").strip()
            prompt = text_without_mention
        else:
            text_without_mention = message.caption.replace(mention, "").strip()
            prompt = text_without_mention
        await ollama_request(message, prompt)

async def process_image(message):
    image_base64 = ""
    if message.content_type == "photo":
        image_buffer = io.BytesIO()
        await bot.download(message.photo[-1], destination=image_buffer)
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
    return image_base64

async def add_prompt_to_active_chats(message, prompt, image_base64, modelname):
    async with ACTIVE_CHATS_LOCK:
        if ACTIVE_CHATS.get(message.from_user.id) is None:
            ACTIVE_CHATS[message.from_user.id] = {
                "model": modelname,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": ([image_base64] if image_base64 else []),
                    }
                ],
                "stream": True,
            }
        else:
            ACTIVE_CHATS[message.from_user.id]["messages"].append(
                {
                    "role": "user",
                    "content": prompt,
                    "images": ([image_base64] if image_base64 else []),
                }
            )

async def handle_response_2(message, full_response, query_duration):
    full_response_stripped = full_response.strip()
    if full_response_stripped == "":
        return
    text = f"{full_response_stripped}\n\n⚙️ {modelname}\nGenerated in {query_duration}s."
    await send_response(message, text)
    async with ACTIVE_CHATS_LOCK:
        if ACTIVE_CHATS.get(message.from_user.id) is not None:
            ACTIVE_CHATS[message.from_user.id]["messages"].append(
                {"role": "assistant", "content": full_response_stripped}
            )
    logging.info(
        f"[Response]: '{full_response_stripped}' for {message.from_user.first_name} {message.from_user.last_name}"
    )
    return True

async def send_response(message, text):
    # A negative message.chat.id is a group message
    if message.chat.id < 0 or message.chat.id == message.from_user.id:
        await bot.send_message(chat_id=message.chat.id, text=text)
    else:
        await bot.edit_message_text(
            chat_id    = message.chat.id,
            message_id = message.message_id,
            text       = text
        )

def ollama_chat(user_input, system_message, ollama_model, conversation_history):
    global vault_embeddings_tensor
    # Load Vault Content
    logging.info("Load Vault")
    vault_content = load_vault(my_filename)
    # Generate Embeddings
    if vault_embeddings_tensor is None:
        logging.info("Generate Tensors Embeddings")
        vault_embeddings_tensor = generate_embeddings(vault_content)
    else:
        logging.info("Using cached embeddings")
    # Get relevant context from the vault
    logging.info("Get Relevant Context from Vault")
    relevant_context= get_relevant_context(user_input, vault_embeddings_tensor, vault_content, top_k=3)
    if relevant_context:
        # Convert list to a single string with newlines between items
        context_str = "\n".join(relevant_context)
        logging.info("Context Pulled from Documents (shown in DEBUG)")
        logging.debug(context_str)
    else:
        logging.warning("No relevant context found.")
    # Prepare the user's input by concatenating it with the relevant context
    logging.info('Prepare User Input')
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input
    # Append the user's input to the conversation history
    logging.info("Append User to History")
    conversation_history.append({"role": "user", "content": user_input_with_context})
    # Create a message history including the system message and the conversation history
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    # Send the completion request to the Ollama model
    logging.info("Send Completion")
    tic = time.perf_counter()
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages
    )
    toc = time.perf_counter()
    logging.info(f"Duration : {toc - tic:0.4f} seconds")
    # Append the model's response to the conversation history
    logging.info("Append Assistant to History")
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    # Return the content of the response from the model
    logging.info("FUNC : End of ollama_chat")
    full_response = response.choices[0].message.content
    query_duration = f"{toc - tic:0.4f}"
    return full_response, query_duration

async def ollama_request(message: types.Message, prompt: str = None):
    try:
        if prompt is None:
            prompt = message.text or message.caption
        
        logging.info(f"[OllamaAPI]: Processing '{prompt}' for {message.from_user.first_name}")
        
        await send_response(message, 'Reçu, en cours de réponse...')
        full_response = ""
        await bot.send_chat_action(message.chat.id, "typing")
        
        image_base64 = await process_image(message)
        await add_prompt_to_active_chats(message, prompt, image_base64, modelname)
        payload = ACTIVE_CHATS.get(message.from_user.id)
        logging.info(f"Payload : {payload}")

        full_response, query_duration = ollama_chat(prompt, system_message, modelname, conversation_history)
        await handle_response_2(message, full_response, query_duration)

    except Exception as e:
        print(f"-----\n[OllamaAPI-ERR] CAUGHT FAULT!\n{traceback.format_exc()}\n-----")
        await bot.send_message(
            chat_id=message.chat.id,
            text=f"Something went wrong.",
            parse_mode=ParseMode.HTML,
        )

async def main():
    await bot.set_my_commands(commands)
    await dp.start_polling(bot, skip_update=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise SystemExit
