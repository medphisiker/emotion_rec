Проект сервиса для определения эмоций человека по видеосвязи (по мимике лица из видео и голосу из аудио).

# Идея сервиса

Я хотел сделать сервис распознавания эмоций людей по видеосвязи (например, для Skype, Zoom и аналогичных сервисов) или телефонных звонков.

Ниже представлен вариант того как может выглядеть сервис(Это красивая визуализация, реализации в коде пока нет).

![Видение_сервиса](repo_pics/Видение_сервиса.png)

Данный сервис может быть полезен всем кто общается со своими клиентами по видео звонкам.
Например, это может быть сотрудник технической поддержки банка дающий онлайн консультацию клиенту банка.
Или преподаватель частной онлайн школы, например, Skillbox. (Например, чтобы видеть радуются ли его студенты, или они невероятно напуганы формулами на доске и им страшно и грустно. Значит пора что то делать).

Данный сервис может помочь оценить эмоциональное состояние как клиента, что поможет в оценке его удовлетворенности обслуживанием, так и контролем работы сотрудника(например, чтобы засечь агрессивное или неподобающее поведение).

# Демо работы веб-сервиса

Ссылка на запущенное веб-приложение ([ссылка](https://medphisiker-emotion-streamlit-app-streamlit-start-n8kbdm.streamlit.app/)).

Ссылка на репозиторий для веб-приложения ([ссылка](https://github.com/medphisiker/emotion_streamlit_app)).

# Продукт для бизнеса

## Польза для бизнеса
Для многих видов бизнеса важны такие показатели KPI как:
* Индекс лояльности клиента — Net Promoter Score(NPS)
* Оценка удовлетворенности клиентов — Customer Satisfaction Score (CSAT)

Так же важно искать лучших специалистов по взаимодействию с клиентами.
Это позволит сформировать сильный и клиенториентированный коллектив.

## Продукт для бизнеса
В лекции про MVP и POC была интересная рекомендация не ограничивать фантазию своих потенциальных клиентов по возможностям использования вашего сервиса в одном направлении.

Я решил вывести этот совет на максимум.

Форма бизнеса - `"Функция как сервис"(function-as-a-service, FaaS)`.
Наш продукт это `docker-образ` с нейронной сетью для распознавания эмоций и удобным `API` для ее использования.
Будет `API` для отправки видео или аудио на сервер и оценки эмоций людей в них.
Данный контейнер разворачивается на серверах и предоставляет заказчику функцию оценки эмоций, которую он может интегрировать в свои сервисы, например, видео чат, или анализ записей звонков от клиентов в тех поддержку.

## Взаимодействие с бизнесом
Находим компанию-клиента заинтересованную в подобном функционале.
Устанавливаем испытательный период, компания заказывает небольшое количество серверов с нашей "Функцией как сервис" и тестово встраивает данную функцию в свои приложения.
Оценивает качество, удобство функции и по своему желанию расширяет тестовую функцию до полноценной интеграции в свои процессы.

Так мы сосредотачиваемся на разработчике моделей для распознавания эмоций, разработке удобного `API`, документации и контейнеризации.

Так же мы делаем демо web-приложение, например на `Streamlit Cloud`, демонстрирующее наши возможности для привлечения клиентов.
И возможно self-хостинг на небольшом числе серверов(одном имеющемся у меня) для дополнительной демонстрации возможностей представителям кампании клиента при личном видео звонке.

Развертывание нашей "Функции как сервиса" может происходить как на собственных серверах компании-клиента(например, `Сбера`), так другой компании подрядчика, например, `Selectel`.

## Варианты получения данных по эмоциям
* Есть вариант статистики и временных интервалов с эмоциями.

Например бизнес хочет узнать не нагрубил ли их сотрудник клиенту и если нагрубил то получить
временные метки и видео этого фрагмента.
Или тестовые зрители смотрят фильм. Клиент снимает их лица и хочет потом получить статистический отчет.

* Есть вариант визуализации эмоций на видео.

Мы получаем видео с оценкой эмоции на нем, как на первой картинке "Видение_сервиса".
Например преподаватель провел лекцию в онлайн сервисе образования, у него есть запись лиц 
каждого из учеников и он хочет посмотреть в режиме записи лекции какие эмоции доставила его лекция слушателям.

# Метрика машинного обучения

Мы хотим принести дополнительную выгоду и пользу бизнесу. Я думаю оценить реальный вклад подобной системы можно по объективным показателям таким как NPS, CSAT и увеличении прибыли компании. Но для этого нужно набирать статистику и анализировать данные полученные до внедрения модели и после. Или проводить A/B тесты с группами где использовалась данная модель и где ее не было.

Сейчас можно сказать, что в задаче распознавания эмоций для бизнеса будет наиболее полезно - точное распознавание эмоций.
Я реши выбрать модель Пола Экмана, в которой выделены отдельные эмоциональные состояния:

![эмоции людей](repo_pics/emotions.jpg)

Задача ставится мной как задача классификации эмоций людей:
* (Реализовано) для видео мы осуществляем детекцию человеческих лиц (я использую готовую модель из `mediapipe`) и отдаем вырезанное лицо и уменьшенное до `224х224` в нейросеть которая обучена на классификацию картинок(эмоций на crop'ах с лицами).

Здесь есть много различных датасетов, но в основном с представителями стран Европы и США. Я думаю, что культурные различия могут накладывать и некоторый bias в интерпретации эмоций и мимики. Но качественных датасетов с представителями из России и стран СНГ мне не удалось найти. Я сделал смелое предположение, что представители стран СНГ выражают свои эмоции так же как и представители стран США и Европы. И что модель обученная на имеющихся датасетах, будет хорошо работать для людей нашей страны.

* (Предстоит реализовать) для аудио мы анализируем звук с помощью нейросети осуществляющей классификацию аудио.
Я планировал использовать dataset для русского языка `Dusha`([ссылка на Хабр](https://habr.com/ru/companies/sberdevices/articles/715468/), [ссылка на paperswithcode](https://paperswithcode.com/dataset/dusha)).

Всего сервис будет распознавать 4 эмоции по видео и аудио:

* Позитив: текст с такой эмоцией необходимо произносить с улыбкой или смехом, стараться делать выраженные ударения на словах, подчеркивающих позитив.

* Нейтраль: прочитать фразу максимально «плоско», не выражая никаких эмоций.

* Грусть: произносить текст с грустью, как будто у тебя был тяжелый рабочий день или приболел хомячок. 

* Злость/раздражение: надо злиться, кричать, негодовать.

Я думаю, что бизнесу важнее находить позитив, и реагировать на негатив. Поэтому другие эмоции не рассматриваются.
(P.S. Мне не удалось найти датасет для аудио с русской речью где были бы и другие эмоции).

В дальнейшем предполагается создать ансамбль и из нейросети обрабатывающей информацию из видео и нейросети обрабатывающей аудио. Часто их объединяют с помощью конкатенации эмбедингов и обучения на этих признаках трансформера.
Однако, для этого нужен датасет с синхронным видео и аудио на русском языке. Такого мне пока не удалось найти.
Я думаю сделать ансамбль объединяющий предсказания данных моделей с опр. весами.
Веса подобрать на данных с синхронным видео с английской речью, а потом просто заменить модель для англииского на модель работающую с русским языком.

Поэтому, я буду использовать метрики для `задачи классификации`.
Как показывает анализ имеющихся датасетов данные по эмоциям сильно не сбалансированы. Ниже представлено распределение по фотографиям с эмоциями из датасета `CelebV-HQ`([link](https://github.com/celebv-hq/celebv-hq)).
![распределение эмоций в датасете CelebV-HQ](repo_pics/statistic.png)

Наиболее частая эмоция, это нейтральное спокойное состояние.
И я думаю, что интерес для бизнеса будут представлять и более сильные, но и более редкие эмоции: например радость или агрессия.
Метрика `Accuracy` нам не подойдет, модель предсказывающая всем нейтральную эмоцию будет права в 70% случаев.
Я думаю, что в данной задаче важны как `precision`, так и `recall` поэтому я решил выбрать в качестве метрики `F1-меру`.

# Текущий dataset

## Исходный датасет
В выбранной задаче анализа видео звонков по web-камере обычно лица людей представлены крупно на экране и человек смотрит на нас анфас.

Для этой задачи подошел датасет `RAVDESS`([link](https://zenodo.org/record/1188976#.ZE6NGSPP2Un)).
В нем 24 профессиональных актера (12 женщин, 12 мужчин), озвучивают две фразы. Актеры произносят речь, которая включает в себя выражения спокойствия, радости, грусти, гнева, страха, удивления и отвращения, и поют данные фразы выражая спокойные, счастливые, грустные, сердитые и испуганные эмоции. 

Каждое выражение создается на двух уровнях эмоциональной интенсивности (нормальном и усиленном).

Пример, того как выглядят исходные видео из данного датасета представлен по данной ссылке на YouTube([link](https://youtu.be/0rvNpbucZOg)). Их видео записи очень похожи на записи сеансов видео связи в Zoom или Skype.

## Предобработка датасета для обучения классификатора эмоций
Я запустил модель детектор лиц из `mediapipe` и нарезал из него набор лиц актеров изображающих различные эмоции
(01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).

Ниже приведен пример полученной картинки из полученного обучающего набора данных:

![эмоции людей](repo_pics/face_crop.png)

Датасет сбалансированный, актеры с порядковыми номерами с 1 по 19 были в выборке `train`(~80%), с 20 по 24 в выборке `test`(~20%).

# Текущая модель (baseline)
На данных лицах была обучена модель классификатор картинок на базе `Resnet50`.
Данный `baseline` представлен в `face_emotion_classifier.ipynb`.

# Результат 
Ниже представлена полученная метрика F1 на датасете `RAVDESS`([link](https://zenodo.org/record/1188976#.ZE6NGSPP2Un)) в задаче классификации эмоций по вырезанным лицам актеров из кадров видео.

```
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Validate metric           DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
         val_f1             0.45323309302330017
        val_loss            2.1491947174072266
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
Ниже представлена `confusion_matrix`, со значениями нормированными на `true` в %.
![confusion_matrix](repo_pics/confusion_matrix.png)

Результаты, пока не очень.
Так и должно быть согласно научным статьям.
Я сразу обучаю модель обученную на `ImageNet` без претрейна на лицах.

Тут бы очень пригодились датасеты:
* AffectNet ([link](http://mohammadmahoor.com/affectnet/))
* Aff-Wild2 ([link](https://www.ibug.doc.ic.ac.uk/resources/aff-wild2/))

Они содержат большее количество обучающих примеров, более обширную выборку людей и их эмоции были выражены естественно.
К сожалению, для доступа к ним требуется быть студентом или научным сотрудником и обладать официальным университетским `email`.
Получить к ним доступ, - я не могу =)

