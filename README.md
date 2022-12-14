# TURBO3000_TOSQL_TRANSLATOR

# Решение 1

Для использования программы необходимо скачать модель с сайта https://rusvectores.org/ru/models/
Обязательно, чтобы модель поддерживала тегсет: "Universal Tags", иначе ничего работать не будет

Перед запуском программы необходимо доустановить библиотеки:
```
pip install gensim
pip install pymorphy2
pip install scipy
pip install json
```
После запуска потребуется указать:
1. Название таблицы на английском
2. Путь до модели.bin
3. Путь до .json файла

Далее можно вводить команды и радоваться/не радоваться

# Cars
Также мы придумали очень простенькую табличку и json файл к ней, на нем можно протестировать нашу программу

![image](https://user-images.githubusercontent.com/62559964/201500228-f842714a-1931-412c-acdb-dffde58509a2.png)

# Решение 2

Параллельно с основным решением мы также развиваем экспериментальный подход, основанный не на ручном задании правил, а на обучении нейросети выделять правила самостоятельно. Для этого ей необходимы примеры обращение к базе данных на русском языке и соответствующий SQL запрос. Чтобы попробовать нашу разработку, вам необходимо запустить блокнот в колабе, загрузить в рабочее пространство пример датасета и модель векторного пространства для русского языка Navec (https://github.com/natasha/navec). Последовательно запустив все ячейки в блокноте, у вас появляется обученная нейронная сеть, способная выдавать SQL-запросы

![Снимок](https://user-images.githubusercontent.com/45196253/201511319-02556e88-aa79-42e0-aa5e-cb59051e8ade.JPG)
