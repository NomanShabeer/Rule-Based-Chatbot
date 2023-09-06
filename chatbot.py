

import nltk
nltk.download('wordnet')

# Set the NLTK data path
nltk.data.path.append("/path/to/your/nltk_data_directory")

# Now, you can download 'punkt' data
nltk.download('punkt')

import nltk
import json
import pickle
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load words, classes, and the chatbot model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    return [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)]

def bag_of_words(sentence):
    return np.array([1 if word in clean_up_sentence(sentence) else 0 for word in words])

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.25
    results = [(i, r) for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[i], 'probability': str(r)} for i, r in results]

def get_response(intents_list, intents_json, context):
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']
            response = random.choice(responses)
            if 'context_set' in intent:
                context['context'] = intent['context_set']
            if 'personalize' in intent:
                user_name = context.get('user_name', 'user')
                response = response.replace("[User's Name]", user_name)
            return response
    return "I'm not sure how to respond to that."

def chat():
    print("GO! Bot is running!")

    conversation_history = []
    context = {}

    while True:
        message = input("You: ")
        conversation_history.append({"user": message})

        if message.lower() == 'quit':
            break

        intents = predict_class(message)
        response = get_response(intents, json.loads(open('intents.json').read()), context)
        conversation_history.append({"bot": response})

        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()

