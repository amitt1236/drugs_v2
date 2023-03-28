from transformers import AutoTokenizer, AutoModel
from torch_geometric.loader import DataLoader
from gnn_model import GNNEncoder
from data import from_smiles
import pandas as pd
import numpy as np
import scipy
import torch

def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return i
        
def validate(dataset, gnn_model, text_model, text):
    tokens = tokenizer(text , add_special_tokens=True, padding='max_length', truncation=True, max_length=77 ,return_tensors="pt", return_attention_mask=True)
    attention_mask = tokens.attention_mask
    tokens = tokens["input_ids"]
    with torch.no_grad():
        text_features = text_model(tokens)
        last_hidden_state = text_features.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask 
    
    
    loader = DataLoader(dataset, batch_size=9600, shuffle=True)
    for batch in loader:
        with torch.no_grad():
            output = gnn_model(batch)
    order = np.array([scipy.spatial.distance.cosine(i, mean_embeddings) for i in output], dtype=np.float32)
    stacked = np.concatenate(((np.expand_dims(order, axis=1), np.expand_dims(np.array(batch.y), axis=1))), axis=1)
    stacked = stacked[np.array(stacked[:, 0], dtype=np.float32).argsort()]
    lst =[df[df['Name']==i[1]].index[0] for i in stacked]

    top100 = np.array(lst[:500])
    same = np.count_nonzero(top100[top100<500])
    print(index_2d(stacked, 'SU3327'))
    print(same)
if __name__ == '__main__':
    # Data
    df = pd.read_csv('./val_data/val.csv')
    df = df[['Name','SMILES']]
    dataset = [from_smiles(row['Name'], row['SMILES']) for idx, row in df.iterrows()]
    
    # gnn model
    gnn_model = GNNEncoder(9, 256, 768, 'GAT')
    gnn_model.load_state_dict(torch.load('/Users/amitaflalo/Desktop/drugs_v2/gnn3500.pth', map_location=torch.device('cpu')))
    gnn_model.eval()

    # text model
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    biobert = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')
    biobert.eval()


    
    # text = "penicillin derivative used for the treatment of infections caused by gram-positive bacteria,\
    # in particular streptococcal bacteria causing upper respiratory tract infections."
    # text = "treat or prevent some types of bacterial infection."
    text = "treat or prevent some types of bacterial infection."

    validate(dataset, gnn_model, biobert, text)