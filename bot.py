#!/usr/bin/env python3

import random
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from zipfile import ZipFile


def get_intent(text):
    probas = clf.predict_proba(vectorizer.transform([text]))[0]
    proba = max(probas)
    if proba > 0.3:
        index = list(probas).index(proba)
        return clf.classes_[index]


# probas = []
# def get_intent(text):
#     proba = clf.score(vectorizer.transform([text])[0], [text])
#     probas.append(proba)
#     print(proba, probas)
#     proba = max(probas)
#     if proba > 0.3:
#         index = list(probas).index(proba)
#         return clf.classes_[index]
# #     intent = clf.predict(vectorizer.transform([text]))[0]
# #     return intent


def get_generative_response(text):
    text = text.lower()
    text = ''.join(char for char in text if char in alphabet)
    text = text.strip()
    words = text.split(' ')

    qas = []
    for word in words:
        if word in search_structure:
            qas += search_structure[word]

    for question, answer in qas:
        if abs(len(text) - len(question)) < len(question) * 0.20:
            edit_distance = nltk.edit_distance(text, question)
            if edit_distance / len(question) < 0.20:
                return answer


def get_response_by_intent(intent):
    phrases = BOT_CONFIG['intents'][intent]['responses']
    return random.choice(phrases)


def get_phailure_phrase():
    phrases = BOT_CONFIG['failure_phrases']
    return random.choice(phrases)


def go_bot(text):
    """Генерация ответной реплики"""
    # NLU
    intent = get_intent(text)

    # Generate answer

    # rules
    if intent:
        stats['intent'] += 1
        return get_response_by_intent(intent)

    # use generative model
    response = get_generative_response(text)
    if response:
        stats['generative'] += 1
        return response

    # stub
    stats['stub'] += 1
    return get_phailure_phrase()


def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')


def help_command(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def bot_answer(update, context):
    """Echo the user message."""
    question = update.message.text
    answer = go_bot(question)
    # print(question, answer)
    # print(stats)
    # print()
    update.message.reply_text(answer)


def main():
    """Start the bot."""
    updater = Updater("1155028770:AAELo8cjnTp0fB4hAMjYgzO5oVnW1DshvSM", use_context=True)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, bot_answer))

    # Start the Bot
    updater.start_polling()
    updater.idle()


with open('BOT_CONFIG.txt', encoding='utf-8') as conf:
    BOT_CONFIG = eval(conf.read())

dataset = []

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        dataset.append([example, intent])

X_text = [x for x, y in dataset]
y = [y for x, y in dataset]

vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))  # Как улучшить?
X = vectorizer.fit_transform(X_text)

clf = LogisticRegression()
# vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5))
# X = vectorizer.fit_transform(X_text)

# clf = LinearSVC()
clf.fit(X, y)

with ZipFile('dialogues.zip') as file:  # read zip
    with file.open('dialogues.txt') as txt:
        dialogues_data = txt.read().decode('utf-8')

dialogues = [dialogue.split('\n')[:2] for dialogue in dialogues_data.split('\n\n')]
dialogues = [dialogue for dialogue in dialogues if len(dialogue) == 2]

dialogues_filtered = []
alphabet = '1234567890- абвгдеёжзийклмнопрстуфхцчшщъыьэюяqwertyuiopasdfghjklzxcvbnm'

for dialogue in dialogues:
    question = dialogue[0][2:].lower()
    question = ''.join(char for char in question if char in alphabet)
    question = question.strip()
    answer = dialogue[1][2:].strip()
    if question and answer:
        dialogues_filtered.append((question, answer))

dialogues_filtered = list(set(dialogues_filtered))

search_structure = {}  # {word: [(q, a), (q, a), ...], ...}

for question, answer in dialogues_filtered:
    words = question.split(' ')
    for word in words:
        if word not in search_structure:
            search_structure[word] = []
        search_structure[word].append((question, answer))

to_del = []
for word in search_structure:
    if len(search_structure[word]) > 10000:
        to_del.append(word)

for word in to_del:
    search_structure.pop(word)

stats = {'intent': 0, 'generative': 0, 'stub': 0}

main()
