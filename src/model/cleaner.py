import pandas as pd
import os
import logging
import contractions
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer
import warnings
warnings.filterwarnings('ignore')

# configure logger
logging.basicConfig(level=logging.INFO,  # Set the logging level
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)',  # Format with filename and line number
                    handlers=[logging.StreamHandler()])  # Log to the console


class Cleaner:
    
    def __init__(self,
                 tokenizer) -> None:
        """
        Constructor
        """
        self.tokenizer = tokenizer
    
    def clean(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        try:
            data_raw = pd.read_csv(f"{os.getcwd()}/results.csv",
                                   delimiter="|")
        except FileNotFoundError:
            logging.error(f"File not found... please put the results.csv file in the right location")
            return None
        
        data_copy = data_raw # df to change 
        
        logging.info("Attempting to clean text data")
        
        data_copy.columns = data_copy.columns.str.replace(' ', '')
        

        # fix this later (automate it)
        data_copy.loc[19999, 'image_name'] = '2199200615.jpg'
        data_copy.loc[19999, 'comment_number'] = '4'
        data_copy.loc[19999, 'comment'] = 'A dog runs across the grass.'
        
        assert len(data_copy['image_name'].unique().tolist()) * 5 == len(data_copy) # assert there is 5 comments per image
        assert not data_raw.isnull().values.all()
        assert 'comment' in data_copy.columns
        
        # TODO ASSERT ALL OTHER INITIAL COLUMNS
        
        return data_copy

    def preprocess_comment(self,
                           comment: str) -> str:
        comment = comment.strip()
        comment = comment.translate(str.maketrans('', '', string.punctuation))
        comment = contractions.fix(comment)
        comment = comment.lower()
        words = word_tokenize(comment)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in stop_words]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        processed_comment = ' '.join(words)
        return processed_comment
    
    def generate_masks_ids(self,
                           comment: str,
                           max_len: int):
        """
        Args:
            comment (str): The comment to tokenize.
            max_len (int): The maximum length for padding/truncation.
        """
        tokens = self.tokenizer.encode_plus(comment,
                                            max_length=max_len,
                                            truncation=True,
                                            padding='max_length',
                                            return_tensors='pt')
            
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        return input_ids, attention_mask


    def find_max_length(self,
                        df: pd.DataFrame):
        sentences = df['cleaned_comment'].values
        max_len = 0
        for i in sentences:
            input_ids = self.tokenizer.encode(i, add_special_tokens=True)
            max_len = max(max_len, len(input_ids))
        return max_len
        
        
    def feature_engineer(self,
                         df: pd.DataFrame,
                         sample_size: int) -> pd.DataFrame:
        """
        asdf

        Args:
            df (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = df[:sample_size] # cut the data into an easy sample size to work with for testing
        logging.info(f"Cut the data into a sample size of {sample_size}")
        logging.info(f"Attempting feature engineering")
        
        df['cleaned_comment'] = df['comment'].apply(self.preprocess_comment)
        assert 'cleaned_comment' in df.columns
        
        results = df['cleaned_comment'].apply(lambda comment: self.generate_masks_ids(comment, self.find_max_length(df=df)))
        df['input_ids'], df['attention_mask'] = zip(*results)
        assert 'input_ids' and 'attention_mask' in df.columns
   
        return df
        
        
# testing
cleaner = Cleaner(tokenizer=BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True))
df = cleaner.clean()
df = cleaner.feature_engineer(df=df, sample_size=20)
print(df)