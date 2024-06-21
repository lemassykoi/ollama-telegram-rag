# For Windows (not tested on Linux)
## You need Ollama

## Then :
* Adjust VARS inside script<br/>
* Start ```1.upload.py``` to create a TXT file which will be parsed for RAG<br/>
* After successful uploading, start ```run.py```<br/>

## Or :
* Adjust VARS inside .env<br/>
* Start ```run_aio.py```<br/>
* ```/help```<br/>
* Send a document on Telegram Chat<br/>
<br/>

## README - Ollama Telegram Bot Script

## Overview
This Python script is designed to create an interactive Telegram bot that interacts with the Ollama API. The bot can handle text messages and documents such as PDFs, TXT, CSV, HTML, and JSON files for processing. It provides functionalities like RAG (Retrieval-Augmented Generation) mode and document embedding.

## Features
- Interaction with Telegram users through text messages and document uploads.
- Handling various document types such as PDF, TXT, CSV, HTML, and JSON.
- Retrieval-Augmented Generation (RAG) mode for contextual understanding of queries.
- Document embedding feature for efficient processing of documents.

## Prerequisites
Before running the script, ensure you have the following:
- Python 3.6 or above.
- A Telegram bot token and a valid username.
- An active Ollama server with appropriate models configured.
- Libraries specified in `requirements.txt` installed.

## Installation & Usage
1. Clone the repository to your local machine:
   ```sh
   git clone https://github.com/lemassykoi/ollama_telegram_bot.git
2. Navigate to the project directory:
   ```sh
   cd ollama_telegram_bot
3. Install required Python packages:
   ```sh
   pip install -r requirements.txt
4. Open `.env` and fill in your Telegram bot token, username, and other necessary details.
5. Run the script using the following command:
   ```sh
   python run_aio.py

## Contributing
Contributions are welcome! If you have any improvements or new features to suggest, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- Thanks to the creators of Ollama and Telegram APIs for providing excellent tools.
- Inspiration from various open-source projects in the AI and telegram bot communities.
<br/>
Forked from https://github.com/AllAboutAI-YT/easy-local-rag and https://github.com/ruecat/ollama-telegram
