# financial-language-model
___
__Language__: English, [Русский](rus.md)

## Description
This repository represents a project aimed at developing an NLP model specialized in the financial domain. The model was fine-tuned on the pre-trained GPT-2 model using diverse data sources, including Reddit, Wikipedia, Investopedia, and financial books. A significant portion of the data was obtained through parsing financial resources. The model was trained using the English language.

__Project Development Objectives__:
* __Studying Text Generation Methods__: The project aims to explore and comprehend text generation methods using NLP models.
* __Training the Model on Financial Data__: The project involves training on specific financial data, which presents its own technical and linguistic complexities.
* __Performance Evaluation__: A crucial part of the project is assessing the model's performance post-training and identifying potential limitations.

## Deployed Demo Model
The model can be tested via the [HuggingFace Space](https://huggingface.co/spaces/kowalsky/financial_wisdom_from_reddit)

## Training Process
__Model and Tokenizer Selection__:

For training, the GPT-2 model and tokenizer were chosen due to the model's capability to perform a wide array of natural language tasks, including text generation. Additionally, GPT-2 offers parameter tuning and the creation of various model configurations.
```Python
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
```
__Data preparation__:

The train_dataset, validation_dataset, and test_dataset have been loaded. The following parameters were used for model training:
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
__Detection and Resolution of Issues__:

During the training process, an issue of overfitting arose, where the validation loss began to gradually increase while the training loss decreased. 
|__Steps__|__Training Loss__|__Validation Loss__|
|------------|-------------|-----------|
|6000|3.470900|3.738574|
|6600|3.457300|3.727550|
|7200|3.374900|3.733540|
|7800|3.373100|3.740214|
|8400|3.363600|3.746500|
|------|-------|-------|

The training was manually stopped, and the num_train_epochs was reduced to 3 to prevent overfitting. This issue can be addressed using the _EarlyStoppingCallback()_ function from Hugging Face's Transformers library.

Additionally, it was observed that the training_dataset primarily consisted of Reddit data, while the validation_dataset contained both Reddit and Wikipedia data. This resulted in the model memorizing the Reddit style, which was reflected in the validation loss.

After troubleshooting and retraining, another significant issue emerged. Model training was performed in parts due to limitations in using _Google Colab_ GPU. Testing revealed that after each new training run, the model forgot the previously learned information.

## Addressing Catastrophic Forgetting
### Dropout and Weight Decay

To mitigate the impact of catastrophic forgetting, techniques such as Dropout and Weight Decay (L2 Regularization) were applied:
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

### Elastic Weight Consolidation

The algorithm was introduced in the [paper](https://arxiv.org/pdf/1612.00796.pdf)

```Python
def get_fisher_diag(model, dataset, params, empirical=True):
    fisher = {}
    params_dict = dict(params)
    # Creating an empty dictionary to store the Fisher diagonal
    for n, p in deepcopy(params_dict).items():
        p.data.zero_()
        fisher[n] = p.data.clone().detach().requires_grad_()

    model.eval()
    # Iterating through the dataset while tracking the progress
    dataset = tqdm(dataset, total=len(dataset))

    for batch in dataset:
        input, _, _, target = batch

        input = input.to(device)
        target = target.to(device)

        model.zero_grad()
        output = model(input)
        output = output.logits
        output = output.view(-1, output.size(-1))
        if empirical:
            label = target.view(-1)
        else:
            label = torch.argmax(output, dim=1)
        # Calculating the loss function and performing backpropagation
        cross_entropy_loss = torch.nn.functional.cross_entropy(output, label)
        cross_entropy_loss.backward()
        # Accumulating gradients to estimate the Fisher diagonal
        for n, p in model.named_parameters():
            fisher[n].data += p.grad.data ** 2 / len(dataset)

    fisher = {n: p for n, p in fisher.items()}
    return fisher

def get_ewc_loss(model, fisher, p_old):
    loss = 0
    # Calculation of EWC loss
    for n, p in model.named_parameters():
        _loss = fisher[n] * (p - p_old[n]) ** 2
        loss += _loss.sum()
    return loss
```
Training process:

```Python
# Moving the model to the selected device (GPU/CPU)
model.to(device)

# Obtaining the diagonal of the Fisher matrix
fisher_matrix = get_fisher_diag(model, train_dataloader, model.named_parameters())
# Cloning the previous model parameters
prev_params = {n: p.data.clone() for n, p in model.named_parameters()}

learning_rate = 0.001
ewc_lambda = 0.1

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop over 3 epochs
for epoch in range(3):
    model.train()
    total_loss = 0.

    train_dataloader = tqdm(train_dataloader, total=len(train_dataloader))

    # Iterating through the training data
    for batch in train_dataloader:
        input, _, _, target = batch

        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(input)
        output = output.logits
        output = output.view(-1, output.size(-1))

        label = target.view(-1)

        # Original loss
        ce_loss = torch.nn.functional.cross_entropy(output, label)

        # EWC loss
        ewc_loss = get_ewc_loss(model, fisher_matrix, prev_params)
        # The final loss function for updating parameters
        loss = ce_loss + ewc_lambda * ewc_loss
        loss.backward()
        optimizer.step()

        train_dataloader.set_description(f"Epoch {epoch+1}")
        train_dataloader.set_postfix(loss=loss.item())
        total_loss += loss.item()

    # Updating the Fisher matrix and previous parameters after each epoch
    if epoch < 2:
        fisher_matrix = get_fisher_diag(model, train_dataloader, model.named_parameters())
        prev_params = {n: p.data.clone() for n, p in model.named_parameters()}
```

### Transfer Learning

To alleviate the catastrophic forgetting effect, a decision was made to expand the neural network architecture by introducing additional layers before subsequent training.

Initial neural network structure:
```
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()
          (c_proj): Conv1D()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
```
Modified neural network structure with the introduction of additional layers:

```Python
class CustomModel(torch.nn.Module):
    def __init__(self, pretrained_model, config):
        super(CustomModel, self).__init__()
        self.transformer = pretrained_model
        self.config = config

        self.ffn1 = torch.nn.Sequential(
            torch.nn.Linear(self.config.vocab_size, self.config.n_embd),
            torch.nn.GELU(),
            torch.nn.Linear(self.config.n_embd, self.config.n_embd)
        )
        self.layer_norm1 = torch.nn.LayerNorm(self.config.n_embd)

        self.ffn2 = torch.nn.Sequential(
            torch.nn.Linear(self.config.n_embd, 2*self.config.n_embd),
            torch.nn.GELU(),
            torch.nn.Linear(2*self.config.n_embd, self.config.n_embd)
        )
        self.layer_norm2 = torch.nn.LayerNorm(self.config.n_embd)

        self.Linear = torch.nn.Linear(self.config.n_embd, self.config.vocab_size)

  def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        hidden_states = self.ffn1(outputs.logits)
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.ffn2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)

        logits = self.Linear(hidden_states)

        return logits

    def generate_text(self, input_ids, max_length=50, temperature=0.9, top_k=50, top_p=0.9):
        with torch.no_grad():
            generated_ids = input_ids.clone()

            for _ in range(max_length):
                logits = self(generated_ids)
                logits = logits[:, -1, :] / temperature
                filtered_logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
                predicted_token = torch.multinomial(probabilities, 1)
                generated_ids = torch.cat((generated_ids, predicted_token), dim=-1)
            return generated_ids

    @staticmethod
    def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., :top_k] = 0
        logits.scatter_(1, sorted_indices_to_remove.to(torch.int64), filter_value)
        return logits
```
After testing this model, unsatisfactory results were obtained in the form of generating incoherent text.
### Results

After all the adjustments and improvements, the model showcased some success in generating text within the financial domain. However, the model's performance still remains inadequate.

## Technologies
* __Python__
* __Hugging Face transformers__
* __Pytorch__
## Conclusion
During the development of this project, the following conclusions were drawn:
* __Data Diversity Importance__: Effective text generation model training required not only fine-tuning parameters but also access to a wide variety of data. Data from various sources such as Reddit, Wikipedia, Investopedia, and financial books proved valuable yet insufficient for efficient model training. 
* __Training Management and Overfitting__: A crucial aspect of development was managing the training process. The need to reduce overfitting demanded a careful selection of parameters and methods. This highlights that successful NLP model development requires not only technical expertise but also experience in managing training.
* __Data Segmentation__: Quality segregation of data into training, validation, and test sets was fundamental for understanding model performance. This process aided not only in training the model but also in evaluating its effectiveness.
* __Catastrophic Interference Effect__: The phenomenon of catastrophic forgetting remains an inevitable characteristic of neural network training, necessitating measures to mitigate this effect.
