#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3

import random
import nltk
import pymorphy2
import requests
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from zipfile import ZipFile


# ## Настройка логирования

# In[ ]:


logging.basicConfig(filename='app.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


# In[ ]:


# Рабочий вариант c CalibratedClassifierCV
def get_intent(text):
    probas = clf.predict_proba(vectorizer.transform([text]))[0]
#     print(len(probas))
    proba = max(probas)
#     print(proba)
    index = list(probas).index(proba)
#     print(index)
    intent = clf.classes_[index]

    logging.info('Намерение: {}. Точность определения: {}'.format(intent, proba))

    if proba > 0.25:
        index = list(probas).index(proba)
#         print(index)
        return clf.classes_[index]
    
    
    
# # Рабочий вариант c LinearSVC
# def get_intent(text):
#     probas = clf.decision_function(vectorizer.transform([text]))[0]
#     print((probas))
#     proba = max(probas)
#     print('сумма:',sum(probas))
#     print(proba)
#     index = list(probas).index(proba)
#     print(index)
#     print(clf.classes_[index])
#     print(clf.predict(vectorizer.transform([text]))[0])
    
#     if proba > -0.25:
#         index = list(probas).index(proba)
#         print(clf.classes_[index])
#         return clf.classes_[index]


# # Рабочий вариант с LogisticRegression
# def get_intent(text):
#     proba = clf.predict_proba(vectorizer.transform([text])[0], [text])
#     probas.append(proba)
#     print(proba, probas)
#     proba = max(probas)
#     if proba > 0.3:
#         index = list(probas).index(proba)
#         return clf.classes_[index]


# In[ ]:


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


list_intents_rzhunemogu = ['joke', 'joke_18']            
            
def get_data_from_rzhunemogu(intent):
    if intent == 'joke':
        url = r'http://rzhunemogu.ru/RandJSON.aspx?CType=1'
        res = requests.get(url)
        return res.text.split(':"')[1][:-2]
    elif intent == 'joke_18':
        url = r'http://rzhunemogu.ru/RandJSON.aspx?CType=11'
        res = requests.get(url)
        return res.text.split(':"')[1][:-2]

def get_response_by_intent(intent):
    if intent in list_intents_rzhunemogu:
        response = get_data_from_rzhunemogu(intent)
    else:
        phrases = BOT_CONFIG['intents'][intent]['responses']
        response = random.choice(phrases)
    logging.info('Ответ BOT_CONFIG: {}'.format(response))
    return response

def get_phailure_phrase():    
    phrases = BOT_CONFIG['failure_phrases']
    response = random.choice(phrases)
    logging.info('Ответ заглушкой: {}'.format(response))
    return response


def go_bot(text):
    
    logging.info('Запрос: {}'.format(text))
    
    """Генерация ответной реплики"""
    # NLU
    intent = get_intent(text)

    # Generate answer

    # rules
    if intent:
        logging.info('Генерация подготовленного ответа')
        return get_response_by_intent(intent)

    # use generative model
    response = get_generative_response(text)
    if response:
        logging.info('Генерация ответа из диалогов: {}'.format(response))
        return response

    # stub
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



# ### Подготовка данных для заготовленных ответов

# In[ ]:


# работаю с документом с заготовленными запросами-ответами
with open('BOT_CONFIG.txt', encoding='utf-8') as conf:
    BOT_CONFIG = eval(conf.read())

dataset = []

# для приведения к нормальной форме всех запросов
morph = pymorphy2.MorphAnalyzer()
# для удаления знаков припинания
alphabet = '1234567890- абвгдеёжзийклмнопрстуфхцчшщъыьэюяqwertyuiopasdfghjklzxcvbnm'
# для исключения повторных запросов
examples = []
i = 0

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        # пропускаю все запросы длинной меньше 4
        if len(example) <= 3:
            continue
        example = example.lower() # приведение к нижнему регистру
        example = morph.parse(example)[0].normal_form # приведение к нормальной форме
        example = ''.join(char for char in example if char in alphabet) # удаление знаков препинания
        if example not in examples:
            examples.append(example)
            dataset.append([example, intent])
        else:
            i += 1
            
print('Количество одинаковых запросов в конфигураторе бота:', i)            


# ### Подготовка данных для обучения

# In[ ]:


X_text = [x for x, y in dataset]
y = [y for x, y in dataset]


# ### Выбор модели обучения и качества обучения

# In[ ]:


# вариант 1
# vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))

# вариант 2
# vectorizer = HashingVectorizer(analyzer='char_wb', ngram_range=(2, 3))

# вариант 3
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))


X = vectorizer.fit_transform(X_text) # обучение

probas = []

for i in range(5):
#     clf = LinearSVC() # вариант 1
    # вариант 2
    svm = LinearSVC()
    clf = CalibratedClassifierCV(svm, method='sigmoid')
#     clf = LogisticRegression() # вариант 3
#     clf = SVC(kernel='precomputed') # вариант 4
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf.fit(X_train, y_train)
    proba = clf.score(X_test, y_test)
    probas.append(proba)
    print('.', end = '')

print('\nКачество модели: {:.2%}'.format(sum(probas) / len(probas)))


# ### Окончательное обучение

# In[ ]:


clf.fit(X, y)


# ### Подготовка данных из библиотеки диалогов

# In[ ]:


# читаем все диалоги из архива
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


# In[ ]:


# go_bot('расскажи пошлый анекдот')


# In[ ]:


main()


# In[ ]:




