import pandas as pd
import os
import sys
import logging
import contractions
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')


sys.path.append(os.getcwd())

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
        assert tokenizer is not None, "Tokenizer must be provided"
        self.tokenizer = tokenizer
    
    def clean(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        try:
            data_raw = pd.read_csv(f"{os.getcwd()}/results.csv", delimiter="|")
            logging.info("Attempting to clean text data")
            
            data_copy = data_raw # df to change 
            data_copy.columns = data_copy.columns.str.replace(' ', '')
            
            assert all(col in data_copy.columns for col in ['comment', 'image_name', 'comment_number']), "Required columns are missing"
            assert len(data_copy['image_name'].unique().tolist()) * 5 == len(data_copy)
            
        except FileNotFoundError:
            logging.error("File not found... please put the results.csv file in the right location")
            return None
        except pd.errors.EmptyDataError:
            logging.error("File is empty")
            return None
        except pd.errors.ParserError:
            logging.error("Error parsing the file")
            return None
        
        
        # fix the one error with numerical value
        self.fix_manual_entries(data_copy)
        assert not data_raw.isnull().values.all()
        
        logging.info(f"After initial clean, df shape: {data_copy.shape}")
        return data_copy
    
    def fix_manual_entries(self, df: pd.DataFrame) -> None:
        """Fix specific entries in the DataFrame."""
        df.loc[19999, 'image_name'] = '2199200615.jpg'
        df.loc[19999, 'comment_number'] = '4'
        df.loc[19999, 'comment'] = 'A dog runs across the grass.'

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
        # assert 'cleaned_comment' in df.columns
        
        results = df['cleaned_comment'].apply(lambda comment: self.generate_masks_ids(comment, self.find_max_length(df=df)))
        df['input_ids'], df['attention_mask'] = zip(*results)
        # assert 'input_ids' and 'attention_mask' in df.columns
        
        # generate image path column
        image_root_path = f"{os.getcwd()}/flickr30k_images/"
        df['image_path'] = image_root_path + df['image_name']
        # assert 'image_path' in df.columns
   
        logging.info(f"After feature engineering, df shape: {df.shape}")
        return df