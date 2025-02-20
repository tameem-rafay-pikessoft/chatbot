# Chatbot

## Description

This chatbot is designed to interact with users and perform various tasks, including embedding document processing, managing embeddings in a vector store, and integrating with Google Drive. It utilizes advanced natural language processing techniques to understand user queries and provide relevant responses.

## Features

- **Document Processing**: Automatically processes documents from a specified Google Drive folder.
- **Embedding Management**: Creates, updates, and deletes embeddings in a vector store.
- **Integration with Google Drive**: Seamlessly interacts with Google Drive to fetch and manage documents.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/tameem-rafay-pikessoft/chatbot
cd chatbot
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

Make sure you have the required dependencies installed by running:

```bash
pip install -r requirements.txt
```

### 4. Store the credential file in the code

Create a credentials.json file in the code to access the Google drive

### 5. Run the Chatbot

Start the chatbot by executing:

```bash
python embeddings.py
```

## Contributing

Feel free to submit issues and pull requests to improve this chatbot.

## License

This project is licensed under the MIT License.
