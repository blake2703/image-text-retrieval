from transformers import BertTokenizer
import torch
import sys
import os
import pandas as pd
from model.cleaner import Cleaner
from model.models import Model
import logging

sys.path.append(os.getcwd())


# configure logger
logging.basicConfig(level=logging.INFO,  # Set the logging level
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)',  # Format with filename and line number
                    handlers=[logging.StreamHandler()])  # Log to the console

if not os.path.isfile(f"{os.getcwd()}/data/finalized_df.csv"):
    logging.warning("Cleaned data not found...")
    logging.info("Now running cleaning and training modules")
    cleaner = Cleaner(tokenizer=BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True))
    df = cleaner.clean()
    df = cleaner.feature_engineer(df=df, sample_size=20)

    model = Model(df=df)
    model.train(df=df)
else:
    logging.warning("Cleaned data found... skipping cleaning and training modules")

df = pd.read_csv(f"{os.getcwd()}/data/finalized_df.csv")
