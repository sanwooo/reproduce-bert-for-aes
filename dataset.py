import re
import pandas as pd 
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

SCORE = {
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60),
}

def read_dataset(prompt_id, fold_id, split):

    ds_path = f"./asap-aes/prompt_{prompt_id}/fold_{fold_id}/{split}.tsv"
    df = pd.read_csv(ds_path, sep='\t')
    df = df.rename({
        'essay_set': 'prompt_id',
        'domain1_score': 'score',
    }, axis=1)
    columns_to_use = ['essay_id', 'prompt_id', 'essay', 'score']
    columns_to_drop = set(df.columns) - set(columns_to_use)
    df = df.drop(columns_to_drop, axis=1)
    return Dataset.from_pandas(df)


def read_datasets(prompt_id, fold_id):
    
    ds_train = read_dataset(prompt_id, fold_id, 'train')
    ds_dev = read_dataset(prompt_id, fold_id, 'dev')
    ds_test = read_dataset(prompt_id, fold_id, 'test')

    return DatasetDict({
        'train': ds_train,
        'dev': ds_dev,
        'test': ds_test,
    })

def preprocess_datasets(tokenizer: AutoTokenizer, datasets: DatasetDict):

    def preprocess(example):
        # example
        # keys: 'essay_id', 'prompt_id', 'essay', 'score'
        
        # scale down score to range [0, 1]
        min_score, max_score = SCORE[example['prompt_id']][0], SCORE[example['prompt_id']][1]
        scale_downed_score = float((example['score'] - min_score) / (max_score - min_score))
        #### todo: scaling y_trues in each prompt to a normal distribution
        #### todo: + smoothing to break ties using essay quality heuristics -> make granularity of scoring same across different prompts (e.g., 2-12, 0-3, 0-60)

        # cleaning essay
        essay = example['essay'].strip()
        essay = re.sub(r' +', ' ', essay)
        pattern = r'@(PERSON|ORGANIZATION|LOCATION|DATE|TIME|MONEY|PERCENT|MONTH|EMAIL|NUM|CAPS|DR|CITY|STATE)\d+'
        essay = re.sub(pattern, lambda x: "@" + x.groups()[0], essay)

        encoded = tokenizer(example['essay'], truncation=True)
        example['y_true'] = scale_downed_score
        example['input_ids'] = encoded['input_ids']
        example['attention_mask'] = encoded['attention_mask']

        return example

    return datasets.map(preprocess, remove_columns=['essay']) 

def load_datasets(prompt_id, fold_id, tokenizer):
    datasets = read_datasets(prompt_id, fold_id)
    datasets = preprocess_datasets(tokenizer, datasets)
    return datasets


