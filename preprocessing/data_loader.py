import csv
import ast
from typing import List, Dict, Any

def load_entity_sentiment_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from CSV where columns are:
    doc_id (int), text (str), entity (list[str]), sentiment (dict[str,int])

    Returns a list of records:
    {
        'doc_id': int,
        'text': str,
        'entities': List[str],
        'sentiments': Dict[str, int]
    }
    """
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = int(row['doc_id'])
            text = row['text']
            entities = ast.literal_eval(row['entity'])
            sentiments = ast.literal_eval(row['sentiment'])
            records.append({
                'doc_id': doc_id,
                'text': text,
                'entities': entities,
                'sentiments': sentiments
            })
    return records
