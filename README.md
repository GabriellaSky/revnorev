# To Revise Or Not to Revise

This repository contains the code associated with the following paper:

[To Revise or Not to Revise: Learning to Detect Improvable Claims for Argumentative Writing Support](https://arxiv.org/abs/2305.16799)
by Gabriella Skitalinskaya and Henning Wachsmuth

## Ready-to-use models (Huggingface)

### Suboptimal Claim Detection

- no context only claim `gabski/deberta-suboptimal-claim-detection`
- with parent claim as contextual information `gabski/deberta-suboptimal-claim-detection-with-parent`
- with main thesis context as contextual information `gabski/deberta-suboptimal-claim-detection-with-thesis`

Example usage: 

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("gabski/deberta-suboptimal-claim-detection")
model = AutoModelForSequenceClassification.from_pretrained("gabski/deberta-suboptimal-claim-detection")
claim = 'Teachers are likely to educate children better than parents.'
model_input = tokenizer(claim, return_tensors='pt')
model_outputs = model(**model_input)

outputs = torch.nn.functional.softmax(model_outputs.logits, dim = -1)
print(outputs)
```

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("gabski/deberta-suboptimal-claim-detection-with-thesis")
model = AutoModelForSequenceClassification.from_pretrained("gabski/deberta-suboptimal-claim-detection-with-thesis")
claim = 'Teachers are likely to educate children better than parents.'
thesis = 'Homeschooling should be banned.'
model_input = tokenizer(claim, thesis, return_tensors='pt')
model_outputs = model(**model_input)

outputs = torch.nn.functional.softmax(model_outputs.logits, dim = -1)
print(outputs)
```

### Claim Improvement Suggestion

- no context only claim `gabski/deberta-claim-improvement-suggestion`
- with parent claim as contextual information `gabski/deberta-claim-improvement-suggestion-with-parent`
- with main thesis context as contextual information `gabski/deberta-claim-improvement-suggestion-with-thesis`

Example usage: 

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("gabski/deberta-claim-improvement-suggestion")
model = AutoModelForSequenceClassification.from_pretrained("gabski/deberta-claim-improvement-suggestion")
claim = 'Teachers are likely to educate children better than parents.'
model_input = tokenizer(claim, return_tensors='pt')
model_outputs = model(**model_input)

outputs = torch.nn.functional.softmax(model_outputs.logits, dim = -1)
print(outputs)
```

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


tokenizer = AutoTokenizer.from_pretrained("gabski/deberta-claim-improvement-suggestion-with-parent-context")
model = AutoModelForSequenceClassification.from_pretrained("gabski/deberta-claim-improvement-suggestion-with-parent-context")
claim = 'Teachers are likely to educate children better than parents.'
parent_claim = 'Homeschooling should be banned.'
model_input = tokenizer(claim,parent_claim, return_tensors='pt')
model_outputs = model(**model_input)

outputs = torch.nn.functional.softmax(model_outputs.logits, dim = -1)
print(outputs)
```
## Reproducing results

### Suboptimal Claim Detection

#### General Experiment + Context

To train the SVM model with static or contextual embeddings:
```
python run_svm.py --input_data './data/combined_data.csv' \
--C '1' \
--model 'glove' \
--context 'None' \
--output_dir './output/svm/glove/' \
--exp_setup 'random'
```

```
python run_svm.py --input_data './data/combined_data.csv' \
--C '1' \
--model 'flair' \
--context 'None' \
--output_dir './output/svm/flair/' \
--exp_setup 'random'
```

```
python run_svm.py --input_data './data/combined_data.csv' \
--C '0.1' \
--model 'microsoft/deberta-base' \
--context 'None' \
--output_dir './output/svm/deberta/' \
--exp_setup 'random'
```


To fine-tune transformer-based models:

```
python run_hf.py --input_data './data/combined_data.csv' \
--n_epochs 5 \
--pretrained_model 'microsoft/deberta-base' \
--output_dir './output/ft-deberta/base/' \
--batch_size 4 \
--warmup_steps 10000 \
--lr 1e-5 \
--eval_steps 1000 \
--save_steps 1000 \
--max_seq_len 128 \
--exp_setup 'random'
```

To include contextual information set the context parameter to the name of the column containing the relevant information and increase max_seq_length, for example:

```
python run_svm.py --input_data './data/combined_data.csv' \
--C '0.1' \
--model 'google/electra-base-discriminator' \
--context 'parent_claim' \
--output_dir './output/svm/electra_parent/' \
--exp_setup 'random'
```

```
python run_hf.py --input_data './data/combined_data.csv' \
--n_epochs 5 \
--pretrained_model 'microsoft/deberta-base' \
--output_dir './output/ft-deberta/base/' \
--context 'title' \
--batch_size 16 \
--warmup_steps 10000 \
--lr 1e-5 \
--eval_steps 1000 \
--save_steps 1000 \
--max_seq_len 256 \
--exp_setup 'random'
```

#### Revision Depth

To fine-tune the transformer-based model run:

```
python run_depth_hf.py  --input_data './data/combined_data.csv' \
--n_epochs 2 \
--pretrained_model 'microsoft/deberta-base' \
--context 'None' \
--output_dir './output/deberta-depth/' \
--batch_size 16 \
--warmup_steps 500 \
--lr 1e-5 \
--eval_steps 100 \
--save_steps 100 \
--max_seq_len 128
```

#### Topical Bias

To change setup to cross-category, set the exp_setup argument to ```'cc'``` instead of ```'random'```. 

To train the SVM model with static or contextual embeddings:

```
python run_svm.py --input_data './data/combined_data.csv' \
--C '0.1' \
--model 'microsoft/deberta-base' \
--context 'None' \
--output_dir './output/svm/deberta/' \
--exp_setup 'cc'
```

To fine-tune the transformer-based model:

```
python run_hf.py --input_data './data/combined_data.csv' \
--n_epochs 5 \
--pretrained_model 'microsoft/deberta-base' \
--output_dir './output/ft-deberta/cc/' \
--batch_size 16 \
--warmup_steps 10000 \
--lr 1e-5 \
--eval_steps 1000 \
--save_steps 1000 \
--max_seq_len 128 \
--exp_setup 'cc'
```

### Claim Improvement Suggestion

To perform multiclass classification with *SVM* or *transformers* run:

```    
python run_multiclass_svm.py --input_data './data/combined_data.csv' \
--C '0.1' \
--model 'google/electra-base-discriminator' \
--context 'None' \
--output_dir './output/svm/electra-multi/'
```

```
python run_multiclass_hf.py --input_data './data/combined_data.csv' \
--n_epochs 5 \
--pretrained_model 'microsoft/deberta-base' \
--context 'title' \
--output_dir './output/ft-deberta-multi/' \
--batch_size 16 \
--lr 1e-5 \
--warmup_steps 10000 \
--eval_steps 1000 \
--save_steps 1000 \
--max_seq_len 128
```

### Data
In order to obtain access to the ClaimRev corpus, please reach out to Gabriella Skitalinskaya (email can be found in paper) along with your affiliation and a short description of how you will be using the data. Please let us know if you have any questions.
