import pandas as pd
import os

def save_to_csv(data, csv_path: str):
    """
    Slaat een lijst van dicts op in een CSV-bestand,
    waarbij je nieuwe rijen toevoegt en dubbels (op Link) filtert.
    """
    df = pd.DataFrame(data)
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_new = pd.concat([df_old, df], ignore_index=True)
        # Dubbele listings op basis van 'Link' voorkomen
        df_new.drop_duplicates(subset=["Link"], keep='last', inplace=True)
        df_new.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)
