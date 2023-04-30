Проект сервиса для определения эмоций человека по видеосвязи (по мимике лица из видео и голосу из аудио).

# Идея сервиса

Я хотел сделать сервис распознавания эмоций людей по видеосвязи (например, для Skype, Zoom и аналогичных сервисов) или телефонных звонков.

Данный сервис может быть полезен всем кто общается со своими клиентами по видео звонкам.
Например, это может быть сотрудник технической поддержки банка дающий онлайн консультацию клиенту банка.
Или преподаватель частной онлайн школы, например, Skillbox. (Например, чтобы видеть радуются ли его студенты, или они невероятно напуганы формулами на доске и им страшно и грустно. Значит пора что о делать).

Данный сервис может помочь оценить эмоциональное состояние как клиента, что поможет в оценке его удовлетворенности обслуживанием, так и контролем работы сотрудника(например, чтобы засечь агрессивное или неподобающее поведение).

# Польза для бизнеса

Для многих видов бизнеса важны такие показатели KPI как:
* Индекс лояльности клиента — Net Promoter Score(NPS)
* Оценка удовлетворенности клиентов — Customer Satisfaction Score (CSAT)

Так же важно искать лучших специалистов по взаимодействию с клиентами.
Это позволит сформировать сильный и клиенториентированный коллектив.

# Метрика машинного обучения

Мы хотим принести дополнительную выгоду и пользу бизнесу. Я думаю оценить реальный вклад подобной системы можно по объективным показателям таким как NPS, CSAT и увеличении прибыли компании. Но для этого нужно набирать статистику и анализировать данные полученные до внедрения модели и после. Или проводить A/B тесты с группами где использовалась данная модель и где ее не было.

Сейчас можно сказать, что в задаче распознавания эмоций для бизнеса будет наиболее полезно - точное распознавание эмоций.
Я реши выбрать модель Пола Экмана, в которой выделены отдельные эмоционалльные состояния:

![эмоции людей](repo_pics/emotions.jpg)

Полученная метрика F1 на датасете `RAVDESS`([link](https://zenodo.org/record/1188976#.ZE6NGSPP2Un)) в задаче классификации эмоций по вырезанным лицам актеров из кадров видео.

```
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Validate metric           DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
         val_f1             0.9543136358261108
        val_loss            0.13473621010780334
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
[{'val_loss': 0.13473621010780334, 'val_f1': 0.9543136358261108}]
```
