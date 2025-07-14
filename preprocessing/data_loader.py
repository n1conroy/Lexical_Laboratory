# preprocessing/data_loader.py

import csv

def load_entity_sentiment_data(path):
    """
    Loads data from CSV file with columns: doc_id, text, entity, sentiment_labels
    Returns a list of dicts for each row.
    """
    records = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                record = {
                    "doc_id": row["doc_id"],
                    "text": row["text"],
                    "entity": row["entity"],
                    "sentiment_labels": row["sentiment_labels"].strip().lower()
                }
                records.append(record)
            except KeyError as e:
                print(f"⚠️ Missing column in row: {row} - {e}")
            except Exception as ex:
                print(f"💥 Failed to load row: {row} - {ex}")

    return records
