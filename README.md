# financial-language-model
___
## Описание
Данный репозиторий содержит проект по разработке модели генерации текста, ориентированной на финансовую сферу. Модель была fine-tuned по предобученной модели gpt2 на данных с Reddit, Wikipedia, Investopedia и разных книг по финансам. Основная часть данных была получена парсингом. Финальная версия модели должна была генерировать ответы на базовые финансовые вопросы, но  после 4 GPU часов обучения в Google Colab производительность модели оказалась недостаточно высокой.

__Цели разработки проекта__:
* Изучение методов генераци текста
* Понимание сложностей, возникающих при обучении модели
* Понимание поведения модели, обученной на датасетах разного размера
## Процесс обучения
Первый раунд обучения производился на модели _gpt2_ и _gpt2_ tokenizer:
```Python
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
```
Затем были загружены train_dataset, validation_dataset и test_dataset. Для обучения модели использовался следующий сет параметров:
```Python
training_args = TrainingArguments(
    output_dir = "/content/drive/MyDrive/NLP/model",
    overwrite_output_dir = False,
    num_train_epochs = 5,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 2,
    save_steps = 600,
    save_total_limit = 2,
    logging_dir = "/content/drive/MyDrive/NLP/logs",
    save_strategy = 'steps',
    evaluation_strategy = 'steps',
    eval_steps = 600,
    logging_steps = 100,
    do_train = True,
    do_eval = True,
    load_best_model_at_end = True,
    remove_unused_columns = True
)
```
Проблема возникла во время обучени:
|__Steps__|__Training Loss__|__Validation Loss__|
|------------|-------------|-----------|
|6000|3.470900|3.738574|
|6600|3.457300|3.727550|
|7200|3.374900|3.733540|
|7800|3.373100|3.740214|
|8400|3.363600|3.746500|
|------|-------|-------|

Тогда как training loss уменьшался, validation loss начал постепенно расти, что указывало на overfitting. Так как я не использовал EarlyStoppingCallback() из Hugging Face's Transformers библиотеки, остановить обучение пришлось вручную. После анализа ситуации я решил уменьшить num_train_epochs до 3, так как ovefitting начался на 4-ом epoch. Также я проверил validation_dataset и обнаружил, что training_dataset состоит преимущественно из постов с Reddit, тогда как validation_dataset - частично из статей с Reddit и Wikipedia. Поэтому модель запоминала стиль с Reddit, что нешативно сказывалось на validation loss.

После устранения ошибок, возникла ещё одна серьёзная проблема. Так как я обучал модель частями из-за ограничения использования Google Colab GPU, то после каждого нового обучения модель забывала предыдущую информацию, при том, что модель была fine-tuned к предыдущей:
```Python
model = GPT2LMHeadModel.from_pretrained("/content/drive/MyDrive/NLP/model")
```
Для уменьшения влияния "catastrophic forgetting" я использовал следующие техники: Dropout и Weight Decay (L2 Regularization):
```Python
# Dropout
model.config.attention_dropout = 0.1
model.config.hidden_dropout_prob = 0.1

# Weight Decay
training_args = TrainingArguments(
    ...
    weight_decay = 0.01
)
```
Также я применил Эластичную Консолидацию Весов (EWC). Данная техника позволяет сохранить ранее выученные паттерны при обучении модели на новом датасете:
```Python
# Инициализация EWC переменных
ewc_lambda = 0.1
prev_model = model.state_dict()
fisher_information = {}

for task in tasks:
  # Поставить модель в режим обучения
  model.train()

  # Рассчитать потери к данной task(датасете)
  for step, batch in enumerate(train_dataset):
    inputs, labels = batch
    outputs = model(**inputs)
    task_loss = outputs.loss()

    # Рассчет градиентов
    task_loss.backward()

    # Обновить информацию Фишера
    for name, param in model.named_parameters():
      if name in fisher_information:
        fisher_information[name] += (param.grad.data ** 2).mean()
      else:
        fisher_information[name] = (param.grad.data ** 2).mean()
    
    # Сбросить градиенты
    optimizer.zero_grad()

    # Добавить регуляризацию EWC к функции потерь
    for name, param in model.named_parameters():
      ewc_term = (fisher_information[name] * (param - prev_model[name]).pow(2)).sum()
      task_loss += (ewc_lamda / 2) * ewc_term

    # Выполнить обратное распространение ошибки и оптимизировать
    task_loss.backward()
    optimizer.step()
  
  # Обновить предыдущую модель
  prev_model = model.state_dict()
```

## Демонстрация модели
## Ограничения и вызовы
Моя модель ограничена в производительности из-за ограниченных ресурсов. Однако этот проект был важным для моего обучения, так как я столкнулся с рядом технических и вычислительных вызовов и научился находить решения в условиях ограниченных ресурсов.
## Инструменты и технологии
* Python
* Библиотека машинного обучения
## Демонстрация результатов
