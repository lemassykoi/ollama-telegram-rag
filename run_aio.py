import logging
import os
import io
import re
import json
import time
import torch
import base64
import ollama
import PyPDF2
import aiohttp
import asyncio
import traceback
from colorama import Back, Fore, Style
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters.command import Command, CommandStart
from aiogram.types import Message, CallbackQuery, BotCommand, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiohttp import ClientTimeout
from asyncio import Lock
from functools import wraps
from dotenv import set_key, get_key

class contextLock:
    lock = Lock()

    async def __aenter__(self):
        await self.lock.acquire()

    async def __aexit__(self, exc_type, exc_value, exc_traceback):
        self.lock.release()

# Get content of vars in .env file
dotenv_path             = ".env"
token                   = get_key(dotenv_path, "TOKEN")
ollama_base_url         = get_key(dotenv_path, "OLLAMA_BASE_URL")
if ollama_base_url is None:
    ollama_base_url     = '127.0.0.1'
ollama_port             = get_key(dotenv_path, "OLLAMA_PORT")
if ollama_port is None:
    ollama_port         = '11434'
log_level_str           = get_key(dotenv_path, "LOG_LEVEL")
if log_level_str is None:
    log_level_str       = 'INFO'
modelname               = get_key(dotenv_path, "INITMODEL")
RAG_MODE                = get_key(dotenv_path, "RAG_MODE")
if RAG_MODE is None:
    RAG_MODE            = 'False'
timeout                 = get_key(dotenv_path, "TIMEOUT")
if timeout is None:
    timeout             = 3000
allowed_ids             = list(map(int, get_key(dotenv_path, "USER_IDS").split(",")))
admin_ids               = list(map(int, get_key(dotenv_path, "ADMIN_IDS").split(",")))
log_levels              = list(logging._levelToName.values())
log_format              = "%(asctime)s - %(name)s - %(levelname)s - %(message)s \t (%(filename)s:%(lineno)d)"
#EMBED_MODEL             = 'mxbai-embed-large'
EMBED_MODEL             = 'snowflake-arctic-embed'
oclient                 = ollama.Client(host=ollama_base_url)
ACTIVE_CHATS            = {}
ACTIVE_CHATS_LOCK       = contextLock()
mention                 = None
CHAT_TYPE_GROUP         = "group"
CHAT_TYPE_SUPERGROUP    = "supergroup"
my_filename             = 'vault.txt'
emoji_false             = '❌'
emoji_true              = '✔️'
conversation_history    = []
system_message          = "Vous êtes un assistant, le plus utile et expert dans l'extraction des informations les plus utiles d'un texte donné. Apportez également des informations supplémentaires pertinentes à la requête de l'utilisateur en dehors du contexte donné."
vault_embeddings_tensor = None
set_key(dotenv_path=dotenv_path, key_to_set="RAG_MODE", value_to_set=str(RAG_MODE))

if log_level_str not in log_levels:
    log_level = logging.DEBUG
else:
    log_level = logging.getLevelName(log_level_str)

logging.basicConfig(
    format=log_format,
    level=log_level
)

current_log_level = logging.getLevelName(logging.root.level)
if current_log_level != log_level_str:
    logging.warning(Back.RED + 'CAUTION : LOG LEVEL IS DIFFERENT FROM OS ENV' + Style.RESET_ALL)

## This needs to be here
logger = logging.getLogger(__name__)

logger.info(Fore.CYAN + "Log Level from OS\t: " + Fore.YELLOW + log_level_str + Style.RESET_ALL)
logger.info(Fore.CYAN + "Current Log Level\t: "+ Fore.YELLOW + current_log_level + Style.RESET_ALL)
if RAG_MODE is True:
    logger.info(Back.CYAN + 'RAG Mode\t\t: ' + Style.RESET_ALL + Fore.YELLOW + 'ON' + Style.RESET_ALL)
else:
    logger.info(Fore.CYAN + 'RAG Mode\t\t: ' + Style.RESET_ALL + Fore.YELLOW + 'OFF' + Style.RESET_ALL)
logger.info(Fore.CYAN + 'Model Name\t\t: ' + Fore.YELLOW + modelname + Style.RESET_ALL)
logger.info(Fore.CYAN + 'Vault File Name\t: ' + Fore.YELLOW + my_filename + Style.RESET_ALL)
logger.info(Fore.CYAN + 'Is Vault Present?\t: ' + Fore.YELLOW + str(os.path.isfile(my_filename)) + Style.RESET_ALL)

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
        except KeyboardInterrupt:
            raise SystemExit

def perms_allowed(func):
    @wraps(func)
    async def wrapper(message: Message = None, query: CallbackQuery = None):
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
    async def wrapper(message: Message = None, query: CallbackQuery = None):
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
                logger.info(f"[MSG] {message.from_user.full_name} ({message.from_user.id}) is not allowed to use this bot.")
            elif query:
                if message and message.chat.type in ["supergroup", "group"]:
                    return
                await query.answer("Access Denied")
                logger.info(f"[QUERY] {message.from_user.full_name} ({message.from_user.id}) is not allowed to use this bot.")

    return wrapper

def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    logger.info(Fore.YELLOW + 'FUNC : Get Relevant Context' + Style.RESET_ALL)
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
    logger.info(Fore.YELLOW + 'FUNC : Get Relevant Context : DONE.' + Style.RESET_ALL)
    return relevant_context

logger.info(Fore.GREEN + 'Start...' + Style.RESET_ALL)
bot                  = Bot(token=token)
dp                   = Dispatcher()
start_kb             = InlineKeyboardBuilder()
settings_kb          = InlineKeyboardBuilder()
commands             = [
    BotCommand(command="start", description="Start"),
    BotCommand(command="reset", description="Reset Chat"),
    BotCommand(command="delete", description="Delete Vault File"),
    BotCommand(command="history", description="Look through messages"),
]

start_kb.row(
    InlineKeyboardButton(text="ℹ️ About", callback_data="about"),
    InlineKeyboardButton(text="⚙️ Settings", callback_data="settings"),
)

settings_kb.row(
    InlineKeyboardButton(text="🔄 Switch LLM", callback_data="switchllm"),
    InlineKeyboardButton(text="🔄 RAG  Mode", callback_data="switchrag"),
    #types.InlineKeyboardButton(text="✏️ Edit Prompt", callback_data="editsystemprompt"),
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
        logger.debug('== QUERY')
        logger.debug(content)
        response = oclient.embeddings(model=EMBED_MODEL, prompt=content)
        logger.debug('=== APPEND')
        vault_embeddings.append(response["embedding"])
        logger.debug('=== Append Done')
    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    logger.info(Fore.GREEN + 'Convert to Tensors : Ok' + Style.RESET_ALL)
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
    logger.info(Back.RED + f'START Asked for {message.from_user.full_name}' + Style.RESET_ALL)
    start_message = f"✨ Welcome, <b>{message.from_user.full_name}</b>!\n\nRAG Mode is currently set to : <b>{RAG_MODE}</b>"
    if RAG_MODE == 'False':
        start_message += f" {emoji_false}"
    elif RAG_MODE == 'True':
        start_message += f" {emoji_true}"
        
    await message.answer(
        text                     = start_message,
        parse_mode               = ParseMode.HTML,
        disable_web_page_preview = True,
    )
    await message.answer(
        text                      = 'Pick a choice',
        parse_mode                = ParseMode.HTML,
        disable_web_page_preview  = True,
        reply_markup              = start_kb.as_markup(),
    )

@dp.message(Command("help"))
async def command_help_handler(message: Message) -> None:
    logger.info(Back.RED + f'HELP Asked for {message.from_user.full_name}' + Style.RESET_ALL)
    help_message = "Bonjour, je suis une IA disponible pour répondre à vos questions. Vous pouvez m'envoyer un document (PDF, TXT, CSV, HTML, JSON) que j'embarquerai dans ma base de données, et vous pourrez me poser des questions sur son contenu, après avoir mis le RAG sur ON\nCela vous sera proposé après l'embarquement d'un document."
    await message.answer(
        help_message,
        parse_mode=ParseMode.HTML,
        reply_markup=start_kb.as_markup(),
        disable_web_page_preview=True,
    )

@dp.message(Command("reset"))
async def command_reset_handler(message: Message) -> None:
    logger.info(Back.RED + f'RESET Asked for {message.from_user.full_name}' + Style.RESET_ALL)
    logger.debug('RESET : Check if User is Allowed')
    if message.from_user.id in allowed_ids:
        logger.debug('RESET : Yes')
        logger.debug('RESET : Check if User is in ACTIVE_CHATS')
        if message.from_user.id in ACTIVE_CHATS:
            logger.debug('RESET : Is in ActiveChats ? YES')
            async with ACTIVE_CHATS_LOCK:
                ACTIVE_CHATS.pop(message.from_user.id)
            logger.info(f"Chat has been reset for {message.from_user.full_name}")
            await bot.send_message(
                chat_id=message.chat.id,
                text="Chat has been reset",
            )
        else:
            logger.debug('RESET : Is in ActiveChats ? NO')
            logger.info(Fore.RED + 'RESET : NO' + Style.RESET_ALL)
    else:
        logger.debug('RESET : Is in Allowed IDs ? NO')
        logger.info(Fore.RED + 'RESET : NO' + Style.RESET_ALL)

@dp.message(Command("delete"))
async def command_delete_handler(message: Message) -> None:
    logger.info(Back.RED + f'DELETE Asked for {message.from_user.full_name}' + Style.RESET_ALL)
    logger.debug('DELETE : Check if User is Allowed')
    if message.from_user.id in allowed_ids:
        logger.debug('DELETE : Yes')
        logger.debug('DELETE : Check if User is in admin_ids')
        if message.from_user.id in admin_ids:
            logger.debug('DELETE : Is in admin_ids ? YES')
            delete_vault_kb = InlineKeyboardBuilder()
            delete_vault_kb.row(
                InlineKeyboardButton(text=f"{emoji_true} OUI", callback_data="delete_vault:True"),
                InlineKeyboardButton(text=f"{emoji_false} NON", callback_data="delete_vault:False"),
            )
            logger.info('Asking to User if he wants to delete Vault File')
            await bot.send_message(
                chat_id=message.from_user.id,
                text=f"Do you really want to <b>DELETE</b> the Vault File <code>{my_filename}</code>?",
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=delete_vault_kb.as_markup()
            )
        else:
            logger.debug('DELETE : Is in admin_ids ? NO')
            logger.info(Fore.RED + 'DELETE : NO' + Style.RESET_ALL)
            await bot.send_message(
                chat_id=message.from_user.id,
                text="NOT AUTHORIZED : <b>admin_ids</b>\nOnly Admins are able to delete Vault File at the moment, sorry",
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True
            )
    else:
        logger.debug('DELETE : Is in Allowed IDs ? NO')
        logger.info(Fore.RED + 'DELETE : NO' + Style.RESET_ALL)
        await bot.send_message(
            chat_id=message.from_user.id,
            text="NOT AUTHORIZED : <b>Allowed IDs</b>",
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )

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
async def settings_callback_handler(query: CallbackQuery):
    logger.info(Back.RED + 'Settings : Modify' + Style.RESET_ALL)
    await bot.send_message(
        chat_id=query.message.chat.id,
        text="Choose the right option.",
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
        reply_markup=settings_kb.as_markup()
    )

@dp.callback_query(lambda query: query.data == "switchrag")
async def switchrag_callback_handler(query: CallbackQuery):
    switchrag_builder = InlineKeyboardBuilder()
    switchrag_builder.row(
        InlineKeyboardButton(text="RAG Mode : ON", callback_data="rag_mode:True"),
        InlineKeyboardButton(text="RAG Mode : OFF", callback_data="rag_mode:False")
    )
    await query.message.edit_text("Choisissez :", reply_markup=switchrag_builder.as_markup(),
    )

async def send_TG_message(text, chat_id):
    await bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )

@dp.callback_query(lambda query: query.data.startswith("rag_mode:"))
async def rag_mode_callback_handler(query: CallbackQuery):
    logger.info(f"Query Data : {query.data}")
    RAG_MODE = query.data.split(":")[1]
    logger.info(Back.RED + f'Trying to set RAG Mode on : {RAG_MODE}' + Style.RESET_ALL)
    if RAG_MODE == 'True':
        # check if file vault.txt exist
        if not os.path.exists(my_filename):
            logger.error(f"The Vault File {my_filename} does not exist.\nCan't enable RAG Mode !")
            RAG_MODE = 'False'
            await bot.delete_message(chat_id=query.message.chat.id, message_id=query.message.message_id)
            await send_TG_message("Can't switch RAG Mode because no Vault File yet. Try to send a document first.", query.message.chat.id)
        else:
            logger.info(Back.RED + f'RAG Mode is now : {RAG_MODE}' + Style.RESET_ALL)
            set_key(dotenv_path=dotenv_path, key_to_set="RAG_MODE", value_to_set=str(RAG_MODE))
            await bot.delete_message(chat_id=query.message.chat.id, message_id=query.message.message_id)
    else:
        logger.info(Back.RED + f'RAG Mode is now : {RAG_MODE}' + Style.RESET_ALL)
        set_key(dotenv_path=dotenv_path, key_to_set="RAG_MODE", value_to_set=str(RAG_MODE))
        await bot.delete_message(chat_id=query.message.chat.id, message_id=query.message.message_id)
    await query.answer(f"RAG Mode : {RAG_MODE}")
    await send_TG_message(f"<b>RAG Mode</b> is now set to : <b>{RAG_MODE}</b>", query.message.chat.id)

@dp.callback_query(lambda query: query.data.startswith("delete_vault:"))
async def delete_vault_callback_handler(query: CallbackQuery):
    logger.info("Callback Handler delete_vault:")
    logger.info(f"Query Data : {query.data}")
    do_delete = query.data.split(":")[1]
    if do_delete == 'True':
        logger.info("Callback Handler delete_vault = TRUE")
        if os.path.exists(my_filename):
            logger.info(f"OK : The file {my_filename} is present")
            try:
                os.remove(my_filename)
                logger.info(Fore.GREEN + "Vault File have been deleted successfully" + Style.RESET_ALL)
            except OSError:
                logger.error(Fore.RED + f"Error when trying to delete Vault File {my_filename}" + Style.RESET_ALL)
        else:
            logger.warning(f"The file {my_filename} does not exist")
        await bot.delete_message(chat_id=query.message.chat.id, message_id=query.message.message_id)
        await bot.send_message(
            chat_id=query.message.chat.id,
            text="✔️ <b>Vault File</b> have been deleted successfully",
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        # As there is no Vault File, set RAG_MODE to OFF
        set_key(dotenv_path=dotenv_path, key_to_set="RAG_MODE", value_to_set=str('False'))
        return RAG_MODE == 'False'
    else:
        logger.info("Callback Handler delete_vault = FALSE")
        await bot.delete_message(chat_id=query.message.chat.id, message_id=query.message.message_id)
        await bot.send_message(
            chat_id=query.message.chat.id,
            text="❌ Delete Vault File Canceled.",
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )

@dp.callback_query(lambda query: query.data == "switchllm")
async def switchllm_callback_handler(query: CallbackQuery):
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
            except KeyError:
                modelfamilies = "✨"
            except KeyboardInterrupt:
                raise SystemExit
        switchllm_builder.row(
            InlineKeyboardButton(
                text=f"{modelname} {modelfamilies}", callback_data=f"model_{modelname}"
            )
        )
    await query.message.edit_text(
        f"{len(models)} models available.\n🦙 = Regular\n🦙📷 = Multimodal", reply_markup=switchllm_builder.as_markup(),
    )

@dp.callback_query(lambda query: query.data.startswith("model_"))
async def model_callback_handler(query: CallbackQuery):
    global modelname
    modelname = query.data.split("model_")[1]
    await query.answer(f"Chosen model: {modelname}")

@dp.callback_query(lambda query: query.data == "about")
@perms_admins
async def about_callback_handler(query: CallbackQuery):
    await bot.delete_message(chat_id=query.message.chat.id, message_id=query.message.message_id)
    logger.info(Back.RED + 'ABOUT Asked' + Style.RESET_ALL)
    dotenv_model = get_key(dotenv_path, "INITMODEL")
    if not os.path.exists(my_filename):
        is_vault_present = emoji_false
    else:
        is_vault_present = emoji_true
    if RAG_MODE == 'True':
        rag_mode_state = emoji_true
    elif RAG_MODE == 'False':
        rag_mode_state = emoji_false
    else:
        rag_mode_state = '❓'
    await bot.send_message(
        chat_id=query.message.chat.id,
        text=f"<b>🧠 RAG Mode</b>\nIs Enabled ? {rag_mode_state}\n\n<b>🗊 Vault File</b>\nIs Present ? {is_vault_present}\n\n<b>🦙 Your LLMs</b>\nCurrently using : <code>{modelname}</code>\nDefault in .env : <code>{dotenv_model}</code>\n\n🪪 This project is under <a href='https://github.com/lemassykoi/ollama-telegram-rag/blob/main/LICENSE'>MIT License.</a>\n\n<a href='https://github.com/lemassykoi/ollama-telegram-rag'>⛭ Source Code</a>",
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )

## Convert File to Vault
@dp.callback_query(lambda query: query.data.startswith("embed_doc:"))
async def embed_document_callback_handler(query: CallbackQuery):
    logger.info(f"Query Data : {query.data}")   ## embed_doc:PDF:True|{file_path}  //   embed_doc:PDF:False
    stripped_query = query.data.split(':')
    do_embed = stripped_query[2]
    doc_type = stripped_query[1]
    logger.info(f"Doc Type      : {doc_type}")
    logger.info(f"Do Embed      : {do_embed}")
    if do_embed == 'False':
        logger.info('Embedding DOC : NO')
        await bot.edit_message_reply_markup(chat_id=query.message.chat.id, message_id=query.message.message_id, reply_markup=None)
        await bot.send_message(chat_id=query.message.chat.id, text="Intégration annulée.")
    else:
        logger.info('Embedding DOC : YES')
        await bot.edit_message_reply_markup(chat_id=query.message.chat.id, message_id=query.message.message_id, reply_markup=None)
        await bot.send_message(chat_id=query.message.chat.id, text="Patientez, intégration en cours...")
        file_path = query.data.split('|')[1]
        if doc_type == 'PDF':
            logger.info(f'Embedding PDF: {file_path}')
            # Lire le fichier PDF avec PyPDF2
            with open(file_path, 'rb') as f:
                logger.info('Read PDF')
                reader = PyPDF2.PdfReader(f)
                logger.info('Count PDF Pages')
                num_pages = len(reader.pages)
                logger.info(f"Nombre de pages : {num_pages}")
                text = ''
                for page_num in range(1, num_pages + 1):
                    logger.info(Fore.CYAN + f"Processing page number: {page_num}" + Style.RESET_ALL)
                    page = reader.pages[page_num - 1]
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
                    if len(current_chunk) + len(sentence) + 1 < 8192:  # +1 for the space
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
        elif doc_type == 'TXT':
            logger.info(f'Embedding TXT: {file_path}')
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
        elif doc_type == 'CSV':
            logger.info(f'Embedding CSV: {file_path} - Not Implemented yet')
            import csv
            with open(file_path, newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
                for row in spamreader:
                    print(', '.join(row))
        elif doc_type == 'HTML':
            logger.info(f'Embedding HTML: {file_path} - Not Implemented yet')
        elif doc_type == 'JSON':
            logger.info(f'Embedding JSON: {file_path}')
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
        ## check rag mode and ask if need to change
        if RAG_MODE == 'False':
            await bot.send_message(chat_id=query.message.chat.id, text="Attention, le RAG Mode est sur OFF.")
            switchrag_builder_kb = InlineKeyboardBuilder()
            switchrag_builder_kb.row(
                InlineKeyboardButton(text="RAG ON", callback_data="rag_mode:True"),
                InlineKeyboardButton(text="RAG OFF", callback_data="rag_mode:False")
            )
            await bot.send_message(
                chat_id=query.message.chat.id,
                text="Voulez-vous le switcher ?",
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=switchrag_builder_kb.as_markup()
            )
        else:
            await bot.send_message(chat_id=query.message.chat.id, text="RAG Mode ON, Ok.")

@dp.message(F.content_type.in_({'document'}))
@perms_allowed
async def handle_document(message: Message):
    from_user_id = message.from_user.id
    if message.document is not None:
        logger.info('Received Document')
        file_id   = message.document.file_id
        file_name = message.document.file_name
        file_path = os.path.join('downloads', file_name)
        logger.info(f'File Name : {file_name}')
        logger.info(file_path)
        if message.document.mime_type == 'application/pdf':
            logger.info('Document is PDF')
            doc_type = 'PDF'
        elif message.document.mime_type == 'text/plain':
            logger.info('Document is TXT')
            doc_type = 'TXT'
        elif message.document.mime_type == 'text/csv':
            logger.info('Document is CSV')
            doc_type = 'CSV'
        elif message.document.mime_type == 'text/html':
            logger.info('Document is HTML')
            doc_type = 'HTML'
        elif message.document.mime_type == 'application/json':
            logger.info('Document is JSON')
            doc_type = 'JSON'
        else:
            TG_answer = f'<b>Document is not supported !</b>\nNo Handler for <code>{message.document.mime_type}</code>'
            log_answer = Fore.RED + f'Document is not supported ! No Handler for {message.document.mime_type}' + Style.RESET_ALL
            logger.warning(log_answer)
            await bot.send_message(
                chat_id=from_user_id,
                text=TG_answer,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            return
        embed_document_kb = InlineKeyboardBuilder()
        embed_document_kb.row(
            InlineKeyboardButton(text="✔️ OUI", callback_data=f"embed_doc:{doc_type}:True|{file_path}"),
            InlineKeyboardButton(text="❌ NON", callback_data=f"embed_doc:{doc_type}:False"),
        )
        # Créer le répertoire de téléchargement s'il n'existe pas
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Télécharger le fichier
        await download_file(file_id, file_path)
        # Demander à l'utilisateur s'il souhaite intégrer le PDF
        logger.info('Asking to User if he wants to embed file to Vault')
        await bot.send_message(
            chat_id=from_user_id,
            text=f"Do you want to embed the provided {doc_type} Document?",
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=embed_document_kb.as_markup()
        )
    elif message.text is not None:
        logger.info('Received Text')
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
    else:
        logger.warning('Unsupported Telegram Message received (not Text & not Docuemnt)')
        await bot.send_message(
            chat_id=from_user_id,
            text="Unsupported Telegram Message received",
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )

@dp.message(F.content_type.in_({'text'}))
@perms_allowed
async def handle_message(message: Message):
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

async def handle_response(message, full_response, query_duration):
    await bot.send_chat_action(message.chat.id, "typing")
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
    logger.debug(Fore.MAGENTA + 
        f"[Response]: '{full_response_stripped}' for {message.from_user.full_name}" + Style.RESET_ALL
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
    relevant_context = False
    if get_key(dotenv_path, "RAG_MODE") == 'True':
    #if RAG_MODE:
        logger.info(Fore.YELLOW + 'RAG Mode : ON' + Style.RESET_ALL)
        global vault_embeddings_tensor
        # Load Vault Content
        logger.info("Load Vault")
        vault_content = load_vault(my_filename)
        # Generate Embeddings
        if vault_embeddings_tensor is None:
            logger.info("Generate Tensors Embeddings")
            vault_embeddings_tensor = generate_embeddings(vault_content)
        else:
            logger.info("Using cached embeddings")
        # Get relevant context from the vault
        logger.info("Get Relevant Context from Vault")
        relevant_context= get_relevant_context(user_input, vault_embeddings_tensor, vault_content, top_k=3)
        if relevant_context:
            # Convert list to a single string with newlines between items
            context_str = "\n".join(relevant_context)
            logger.info("Context Pulled from Documents (shown in DEBUG)")
            logger.debug(context_str)
        else:
            logger.warning("No relevant context found.")
    elif get_key(dotenv_path, "RAG_MODE") == 'False':
    #elif not RAG_MODE:
        logger.info(Fore.YELLOW + 'RAG Mode : OFF' + Style.RESET_ALL)
    else:
        logger.error(Fore.RED + 'RAG Mode UNKNOWN' + Style.RESET_ALL)
    # Prepare the user's input by concatenating it with the relevant context
    logger.info('Prepare User Input')
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input
    
    # Append the user's input to the conversation history
    logger.info("Append User to History")
    conversation_history.append({"role": "user", "content": user_input_with_context})
    # Create a message history including the system message and the conversation history
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    # Send the completion request to the Ollama model
    logger.info(Fore.GREEN + "Send Completion" + Style.RESET_ALL)
    tic = time.perf_counter()

    full_response = oclient.chat(model=ollama_model, messages=messages)  ## This full response contains duration
    response = full_response.get('message').get('content')
    toc = time.perf_counter()
    logger.info(Fore.MAGENTA + f"Duration : {toc - tic:0.4f} seconds" + Style.RESET_ALL)
    # Append the model's response to the conversation history
    logger.info("Append Assistant to History")
    conversation_history.append({"role": "assistant", "content": response})
    # Return the content of the response from the model
    logger.info(Fore.GREEN + "FUNC : End of ollama_chat" + Style.RESET_ALL)
    full_response = response
    query_duration = f"{toc - tic:0.4f}"
    return full_response, query_duration

async def ollama_request(message: Message, prompt: str = None):
    try:
        if prompt is None:
            prompt = message.text or message.caption
        
        logger.debug(Fore.MAGENTA + f"[OllamaAPI]: Processing '{prompt}' for {message.from_user.full_name}" + Style.RESET_ALL)
        bot_message = 'Reçu, en cours de réponse...\n'
        
        if RAG_MODE == 'True':
            bot_message += '(avec contexte)'
        else:
            bot_message += '(sans contexte)'
        
        await send_response(message, bot_message)
        await bot.send_chat_action(message.chat.id, "typing")
        
        image_base64 = await process_image(message)
        await add_prompt_to_active_chats(message, prompt, image_base64, modelname)
        payload = ACTIVE_CHATS.get(message.from_user.id)
        logger.debug(f"Payload : {payload}")

        full_response, query_duration = ollama_chat(prompt, system_message, modelname, conversation_history)
        await handle_response(message, full_response, query_duration)

    except Exception as e:
        print(f"-----\n[OllamaAPI-ERR] CAUGHT FAULT!\n{traceback.format_exc()}\n-----")
        await bot.send_message(
            chat_id=message.chat.id,
            text=f"Something went wrong : {e}",
            parse_mode=ParseMode.HTML,
        )
    except KeyboardInterrupt:
        raise SystemExit

# Function to download file from Telegram
async def download_file(file_id, file_path):
    file_info = await bot.get_file(file_id)
    file_url = f'https://api.telegram.org/file/bot{token}/{file_info.file_path}'
    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as resp:
            if resp.status == 200:
                with open(file_path, 'wb') as f:
                    f.write(await resp.read())
            else:
                raise Exception('Failed to download file')

async def main():
    await bot.set_my_commands(commands)
    await dp.start_polling(bot, skip_update=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise SystemExit
