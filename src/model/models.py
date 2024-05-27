import pandas as pd
import torch
import torchvision
from PIL import Image
import os
from transformers import BertModel
import sys
import logging

sys.path.append(os.getcwd())

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    print("No GPU found")
device


logging.basicConfig(level=logging.INFO,  # Set the logging level
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)',  # Format with filename and line number
                    handlers=[logging.StreamHandler()])  # Log to the console

class Model:
    
    def __init__(self,
                 df: pd.DataFrame) -> None:
        self.df = df
        self.vgg_model = torchvision.models.vgg16(pretrained=True).to(device)
        self.vgg_model = torch.nn.Sequential(*list(self.vgg_model.children())[:-1])
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    
    def train(self,
              df: pd.DataFrame):
        df['vgg16_embeddings'] = None
        df['bert_embeddings'] = None
        
        
        assert all(col in df.columns for col in ['comment', 'image_name', 'comment_number', 'image_path', 'input_ids', 'attention_mask', 'cleaned_comment', 'bert_embeddings', 'vgg16_embeddings']), "Required columns are missing"
        logging.info("Attempting to train...")
        
        processor = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

        self.vgg_model.eval()
        self.bert_model.eval()
    
        with torch.no_grad():
            for i in range(len(df)):
                # handle vgg 
                image_path = df.at[i, 'image_path']
                img = Image.open(image_path).convert('RGB')
                img_tensor = processor(img).unsqueeze(dim=0).to(device)
                embeddings = self.vgg_model(img_tensor)
                df.at[i, 'vgg16_embeddings'] = embeddings.cpu().numpy()
            
                # handle bert
                input_ids = df.at[i, 'input_ids'].to(device)
                attention_mask = df.at[i, 'attention_mask'].to(device)
                outputs = self.bert_model(input_ids, attention_mask)
                bert_embeddings = outputs.last_hidden_state
                df.at[i, 'bert_embeddings'] = bert_embeddings.cpu().numpy()
        
        logging.info("Saving data and models...")
        self.df = df
        self.df.to_csv(f"{os.getcwd()}/data/finalized_df.csv")
        torch.save(self.vgg_model.state_dict, f"{os.getcwd()}/data/vgg_16.pth")
        torch.save(self.vgg_model.state_dict, f"{os.getcwd()}/data/bert.pth")
