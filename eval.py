from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, AutoModel
from gnn_model import GNNEncoder
from data import CustomDataset
import plotly.express as px
import pandas as pd
import numpy as np
import torch
import scipy
from sklearn.manifold import TSNE

model = GNNEncoder(9, 256, 768, 'GAT')
model.load_state_dict(torch.load('/Users/amitaflalo/Desktop/drugs_v2/gnn3500.pth', map_location=torch.device('cpu')))
model.eval()

dataset = CustomDataset('./parse_data/sum_smiles.csv')
loader = DataLoader(dataset, batch_size=3600, shuffle=True)
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
biobert = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')
biobert.eval()
tokens = tokenizer('penicillin derivative used for the treatment of infections caused by gram-positive bacteria, in particular streptococcal bacteria causing upper respiratory tract infections.', add_special_tokens=True, padding='max_length', truncation=True, max_length=77 ,return_tensors="pt", return_attention_mask=True)
attention_mask = tokens.attention_mask
tokens = tokens["input_ids"]
with torch.no_grad():
    text_features = biobert(tokens)
    last_hidden_state = text_features.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask 

for batch in loader:
    with torch.no_grad():
        output = model(batch)


    # names, ind = zip(*batch.y)
    # tsne = TSNE(n_components=2)
    # tsne_results = tsne.fit_transform(np.array(output))
    # x_main, y_main = tsne_results[names.index('Halicin')]
    # radius  = 5
    # closest  = [1 if (x[0] < x_main + radius and x[0] > x_main - radius) and (x[1] < y_main + radius and x[1] > y_main - 5) else 0 for x in list(tsne_results)]
    # print(np.array(batch.y)[np.nonzero(np.array(closest))])

    
order = np.array([scipy.spatial.distance.cosine(i, mean_embeddings) for i in output], dtype=np.float32)
stacked = np.concatenate(((np.expand_dims(order, axis=1), batch.y)), axis=1)
stacked = stacked[np.array(stacked[:, 0], dtype=np.float32).argsort()]
print(stacked[:20])