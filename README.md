# CodSoft-_Internship-Rule-Based-Chatbot
# Chatbot using TensorFlow and NLTK

## Introduction

This project demonstrates a chatbot implemented using TensorFlow and NLTK (Natural Language Toolkit). The chatbot is capable of answering questions, providing information, and engaging in conversations with users.

## Features

- **Natural Language Processing (NLP):** The chatbot uses NLP techniques to understand and respond to user input.

- **Intent Recognition:** It recognizes user intents and provides relevant responses.

- **Contextual Understanding:** The chatbot maintains context across conversations for a more natural interaction.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- NLTK
- NumPy

### Installation

1. Clone this repository:
   ```bash
(https://github.com/NomanShabeer/CodSoft-_Internship-Rule-Based-Chatbot.git)
   cd your-repo
   ```

2. Install the required dependencies:
   ```bash
   pip install tensorflow nltk numpy
   ```

3. Download NLTK data (WordNet Lemmatizer and Tokenizer):
   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('punkt')
   ```

## Usage

1. Train the chatbot model:

   ```bash
   python training chatbot.py
   ```

   This will train the chatbot using the intents and responses defined in `intents.json` and save the model as `chatbot_model.h5`.

2. Start the chatbot:

   ```bash
   python chatbot.py
   ```

   You can now chat with the bot. Type "quit" to exit the conversation.

## Intents and Responses

The chatbot's behavior is defined in the `intents.json` file. You can customize the intents and responses to suit your requirements.

## Model

The chatbot model is a neural network trained on the intents and responses data. It uses TensorFlow for training and inference.

## Acknowledgments

- This chatbot was created using TensorFlow and NLTK.

---
