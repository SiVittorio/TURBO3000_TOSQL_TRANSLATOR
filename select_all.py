from gensim import models
import pymorphy2
from scipy import spatial
import pandas as pd
from collections import Counter
import json
from pprint import pprint
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

table_name = input('Введите название таблицы (eng):')

model = models.KeyedVectors.load_word2vec_format(input('Введите путь до бинарной модели:'), binary=True)
morph = pymorphy2.MorphAnalyzer()

#############################################################################
####Оптимизация слов под формат модели gensim и пропуск неизвестных слов#####
def normalizator(words):
    nrml_words = []
    i = 0
    for word in words:
        form_string = morph.parse(word.lower())[0]
        form_string = form_string.normal_form
        mor_string = morph.parse(form_string)[0]
        try:
            nrml_words = nrml_words + [form_string + '_' + mor_string.tag.POS]
            #print(nrml_words)
        except TypeError:
                print(f'{word} - падеж не определен')
                continue
        if ('ё' in nrml_words[i]):
            nrml_words[i] = nrml_words[i].replace('ё', 'е')
            #print(nrml_words[i])
        if ("INFN" in nrml_words[i]):
            #print("INFT TO VERB") 
            nrml_words[i] = nrml_words[i].partition('_')[0] + "_VERB"
            if (nrml_words[i] in model):
                i += 1
            else:
                print('Слова нет в модели')
                nrml_words.pop(i)
 
        elif ("PRCL" in nrml_words[i]):
           # print("PRCL TO NOUN")
            nrml_words[i] = nrml_words[i].partition('_')[0] + "_NOUN"
            if (nrml_words[i] in model):
                i += 1
            else:
                print('Слова нет в модели')
                nrml_words.pop(i)     
        elif ("ADJF" in nrml_words[i]):
            #print("PRCL TO NOUN")
            nrml_words[i] = nrml_words[i].partition('_')[0] + "_NOUN"
            if (nrml_words[i] in model):
                i += 1
            else:
                print('Слова нет в модели')
                nrml_words.pop(i)
        elif ("ADVB" in nrml_words[i]):
           # print("PRCL TO NOUN")
            nrml_words[i] = nrml_words[i].partition('_')[0] + "_NOUN"
            if (nrml_words[i] in model):
                i += 1
        elif ("NUMR" in nrml_words[i]):
           # print("PRCL TO NOUN")
            nrml_words[i] = nrml_words[i].partition('_')[0] + "_NUM"
            if (nrml_words[i] in model):
                i += 1
            else:
                print('Слова нет в модели')
                nrml_words.pop(i)
        elif (False):#("PREP" in nrml_words[i]):
           # print("PRCL TO NOUN")
            nrml_words[i] = nrml_words[i].partition('_')[0] + "_X"
            if (nrml_words[i] in model):
                i += 1
            else:
                print('Слова нет в модели')
                nrml_words.pop(i)
        elif nrml_words[i] in model:
            i += 1
            
        else:
            print('Слова нет в модели')
            nrml_words.pop(i)
    #print(nrml_words)
    return nrml_words

#############################################################################
##################Получение json-файла и конвертация в текст#################

def open_json(file_name):
    with open(file_name, encoding="utf-8") as f:
        words = json.load(f)
    #print(words)
    caption = []
    return words
    
#############################################################################
########################### SELECT * FROM
def select_all_check(string, themes):
    similar_max = 0
    verb_check = False
    columns_max = 0
    for i in range(len(string)):
        if (similar_max < model.similarity('все_NOUN', string[i])):
            similar_max = model.similarity('все_NOUN', string[i])
        if ('VERB' in string[i] and model.similarity('показывать_VERB', string[i]) >= 0.2):
            print(model.similarity('показывать_VERB', string[i]))
            verb_check = True
        ####ПРОВЕРКА НА НАЗВАНИЯ СТОЛБЦОВ
        for columns in themes:
            for word in columns:
                if (columns_max < model.similarity(word, string[i])):
                    columns_max = model.similarity(word, string[i])
    print(similar_max, verb_check, columns_max)
    if (similar_max > 0.1 and verb_check):
        return True, columns_max
    else:
        return False, columns_max
#############################################################################
########################### SELECT COUNT(*) FROM
def select_count_all_check(string, themes):
    similar_max = 0
    columns_max = 0
    for i in range(len(string)):
        if (similar_max < model.similarity('сколько_NOUN', string[i])):
            similar_max = model.similarity('сколько_NOUN', string[i])
        ####ПРОВЕРКА НА НАЗВАНИЯ СТОЛБЦОВ
        for columns in themes:
            for word in columns:
                if (columns_max < model.similarity(word, string[i])):
                    columns_max = model.similarity(word, string[i])
    print(similar_max, columns_max)
    if (similar_max > 0.3):       
        return True, columns_max
    else:
        return False, columns_max
    #Сколько всего столбцов в таблице
    #Сколько полей где код равен одному
#############################################################################
def search_column(string, themes, eng_themes):
    for i in range(len(string)):
        for j in range(len(themes)):
            for k in range(len(themes[j])):
                if (model.similarity(themes[j][k], string[i]) > 0.8):
                    return eng_themes[j], i
#############################################################################
def theme_checker_select(commands):
    mas = []
    for i in range(len(commands)):
        commands[i] = normalizator(commands[i].split())
        mas.append(select_all_check(commands[i]))
    return mas
def theme_checker_count_select(commands):
    mas = []
    for i in range(len(commands)):
        commands[i] = normalizator(commands[i].split())
        print(commands[i])
        mas.append(select_count_all_check(commands[i]))
    return mas
#############################################################################
select_commands = ['Покажи всю таблицу', 'Дай мне всё', 'Выведи все значения', 'Покажи все записи']
count_select_commands = ['Сколько записей в таблице']
test_commands = ['Покажи все записи', 'Сколько уникальных пропусков', 'Сколько ип', 'Сколько записей в таблице']


######Открытие json файла 
columns_names_rus = []
columns_names_eng = []

words = open_json(input('Введите путь до json файла:'))
for cols in words['Columns']: 
    columns_names_rus.append(cols['Caption']) ##получение русских названий столбцов
for cols in words['Columns']: 
    columns_names_eng.append(cols['Name']) ##получение английский названий столбцов
print(columns_names_eng)
#############################################################################
'''
1. Чтение json (✓)
2. Компановщик команд (~)
3. Больше вариантов команд (~)
'''
norm_columns_names_rus = [column.split() for column in columns_names_rus]

for i in range(len(norm_columns_names_rus)):#нормализация названий столбцов
    norm_columns_names_rus[i] = normalizator(norm_columns_names_rus[i])
   
#print(norm_columns_names_rus)

    
#print(theme_checker_select(test_commands))
#print(theme_checker_count_select(count_select_commands))
#############################################################
def sql_writer(command, eng_names, rus_names, table_name):
    norm_command = normalizator(command.split())
    select = select_all_check(norm_command, rus_names)
    print(norm_command)
    if (select[0] and  select[1] < 0.5):
        ####SELECT * FROM
        return ' '.join(('SELECT * FROM', table_name))
    elif (select[0] and select[1] > 0.5):
        ####SELECT * FROM WHERE <>
        search = search_column(norm_command, rus_names, eng_names)
        return ' '.join(('SELECT * FROM', table_name, 'WHERE', search[0].upper(), ''.join(('= ', '\'', command.split()[search[1] + 1], '\''))))

    select = select_count_all_check(norm_command, rus_names)
    print(select)
    if (select[0] and  select[1] < 0.5):
        ####SELECT COUNT(*) FROM
        return ' '.join(('SELECT COUNT(*) FROM', table_name))
    elif (select[0] and select[1] > 0.5):
        search = search_column(norm_command, rus_names, eng_names)
        return ' '.join(('SELECT COUNT(*) FROM', table_name, 'WHERE', search[0].upper(), ''.join(('= ', '\'', command.split()[search[1] + 1], '\''))))
#############################################################       
while (True):
    command = input('Введите команду: ')
    if (command == 'выход'):
        break
    print(sql_writer(command, columns_names_eng, norm_columns_names_rus, table_name))
