# financial-language-model
___
## Описание
Данный репозиторий представляет собой проект, направленный на разработку NLP модели, специализированной в финансовой сфере. Модель была fine-tuned на предобученной модели gpt2 с использованием разнообразных источников данных, включая Reddit, Wikipedia, Investopedia и финансовые книги. Большая часть данных была получена путем парсинга финансовых ресурсов. Модель была обучена на английском языке.

__Цели разработки проекта__:
* __Изучение методов генераци текста__: Проект направлен на исследование и понимание методов генерации текста с использованием NLP моделей.
* __Обучение модели на финансовых данных__: Проект предполагает обучение на специфических финансовых данных, что представляет свои собственные технические и лингвистические сложности.
* __Оценка производительности__: Важной частью проекта является оценка производительности моедли после обучения, а также выявление возможных ограничений.
## Процесс обучения
__Выбор модели и токенизатора__:

Для обучения были выбраны модель и токенизатор _GPT-2_ ввиду способности модели выполнять широкий спектр задач в рамках естественного языка, включая генерацию текста. Также _GPT-2_ предоставляет возможность настройки параметров и и создания различных конфигураций модели.
```Python
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
```
__Подготовка данных__:

Загружены train_dataset, validation_dataset и test_dataset. Для обучения модели использовались следующие параметры:
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
__Обнаружение и решение проблем__:

В процессе обучения возникла проблема с overfitting, когда validation loss начал постепенно расти, в то время как training loss уменьшался. 
|__Steps__|__Training Loss__|__Validation Loss__|
|------------|-------------|-----------|
|6000|3.470900|3.738574|
|6600|3.457300|3.727550|
|7200|3.374900|3.733540|
|7800|3.373100|3.740214|
|8400|3.363600|3.746500|
|------|-------|-------|

Обучение было остановлено вручную, и num_train_epochs был уменьшен до 3, чтобы избежать overfitting. Данную проблему можно устранить функцией EarlyStoppingCallback() из Hugging Face's Transformers библиотеки.

Также, было обнаружено, что training_dataset включал в себя преимущественно данные с Reddit, в то время как validation_dataset содержал как данные с Reddit, так и Wikipedia. Это привело к тому, что модель запомнила стиль Reddit, что отразилось на validation loss.

После устранения ошибок и обучения, возникла ещё одна серьёзная проблема. Обучение модели проводилось частями из-за ограничения использования Google Colab GPU. Тестирование показало, что после каждого нового курга обучения модель забывала предыдущую информацию.

__Борьба с "catastrophic forgetting"__:

Чтобы снизить влияние "catastrophic forgetting", были применены техники Dropout и Weight Decay (L2 Regularization):
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
__Результаты__:

После всех настроек и улучшений, модель продемонстрировала некоторые успехи в генерации текста в финансовой сфере. Однако, производительность модели остается недостаточной.
## Демонстрация модели
При тестировании модели были использованы следующие параметры:
```Python
model_dir = "/content/drive/MyDrive/NLP/final_model"
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def chat_with_model():
  while True:
    user_input = input("You: ")
    input_ids = tokenizer.encode(user_input, return_tensors='pt')

    response_ids = model.generate(input_ids, max_length=100, num_return_sequences=1,
                                  temperature=0.9, top_k=15, do_sample=True,
                                  pad_token_id=model.config.eos_token_id)

    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    print(f"Model: {response}")

chat_with_model()
```
Ниже представлены результаты:
> You: Can you advice me on how to trade?
> > Model: Can you advice me on how to trade? Thanks. Just a few tips and a question :
> > 1
> > You should try to trade in a way that allows you to trade on multiple stocks. A very important rule of trading is to never look at one stock, let only a few stocks appear and then > > move on. I use a single stock as a guide, but if I look at a chart, I will look at the whole chart.
> > 2
> > Don't look for a trade
> You: What is cryptocurrency?
> > Model: What is cryptocurrency?
> > It's a digital currency.
> > You don't need to own a cryptocurrency to use it. Just have a wallet and have access to the network, and you don't need to pay for it.
> You: Explain the concept of hedge fund.
> > Model: Explain the concept of hedge fund. 
> > What is the hedge fund? 
> > The term “hedge fund” is used to describe hedge fund due diligence BLACKLISTING for a person to serve as tireless in the due diligence process of their hedge fund and to make the > > > hedge fund sound and professional for the next two to five years. The hedge fund due diligence process is done by professionals in the hedge fund due diligence program of the hedge > > fund, not by hedge funds.
> What is blockchain?
> > Model: What is blockchain?
> > It’s essentially a system to verify transactions between parties in a blockchain without the need for a central server, or an authority for any of the other things
Неудачные результаты теста:
> You: How to become rich?
> > Model: How to become rich?
> > I'm not a billionaire, I'm a f**ing human being.
> > I am not saying that I am self-made. I'm saying that I am self-made.
> You: How to calculate a stock price?
> > Model: How to calculate a stock price?
> > Stock price = Stock price + Stock price - (100-100)
> > The stock  (stock)  is the stock with the highest number of days to maturity. The market price of the stock (the price at which it is sold) is called the market price.
> You: If I buy stocks without conducting research, will I lose my money?
> > Model: If I buy stocks without conducting research, will I lose my money?
> > It's all bu***hit
> > You know what to watch out for. You’re all idiots who have no idea what they are doing. 
> > If you have the ability to read charts and you are going to watch the news, then you know you are watching what happens next, so you don't lose your money.
## Инструменты и технологии
* Python
* Hugging Face transformers
* Pytorch
## Вывод
В процессе разработки данного проекта были сделаны следующие выводы:
* __Значение разнообразия данных__: Для эффективного обучения модели генерации текста, было не только важно правильно настроить параметры, но и иметь доступ к обширному разнообразию данных. Данные из разных источников, такие как Reddit, Wikipedia, Investopedia и финансовые книги, оказались весьма ценными, но недостаточными для эффективного обучения модели. 
* __Управление обучением и overfitting__: Важной составляющей разработки было управление процессом обучения. Необходимость в снижении overfitting требовала внимательного подхода к выбору параметров и методов. Это демонстрирует, что успешная разработка NLP модели не только требует технической экспертизы, но и опыта в управлении обучением.
* __Разделение данных__: Качественное разделение данных на тренировочные, валидационные и тестовые наборы данных оказалось фундаментальным для понимания производительности модели. Этот процесс помог не только в обучении модели, но и в оценке её эффективности.
* __Эффект catastrophic interference__: Эффект катастрофического забывания является неизбежной особенностью обучения нейронных сетей, что требует принятия мер для уменьшения данного эффекта.
