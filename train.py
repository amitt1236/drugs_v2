from transformers import AutoTokenizer, AutoModel
from torch_geometric.loader import DataLoader
from gnn_model import GNNEncoder
from data import CustomDataset
from tqdm import tqdm
import torch

def train(graph_model, text_model, tokenizer, loader, device, epoch):
    count = 0
    for _  in tqdm(range(epoch)):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            count = count  + 1
            opt.zero_grad()

            # Convert text to tensors
            tokens = tokenizer.batch_encode_plus(batch.y, add_special_tokens=True, padding='max_length', truncation=True, max_length=128 ,return_tensors="pt", return_attention_mask=True)["input_ids"].to(device)
            text_features = text_model(tokens).pooler_output
            
            graph_features = graph_model(batch)
            if text_features.shape[0] != batch.batch.max()+1:
                print(f"graph feats {graph_features.shape[0]}, text feat {text_features.shape[0]} batch {batch.batch.max()+1}")

            labels = torch.arange(batch.batch.max()+1 ,dtype=torch.long, device=device)

            loss = gnn_model.loss(graph_features, text_features, labels)
            print(loss)
            loss.backward()
            opt.step()
            total_loss = total_loss + loss
        
        print(total_loss / len(loader))
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load data
    dataset = CustomDataset('./parse_data/sum.csv')
    loader = DataLoader(dataset, batch_size=36, shuffle=True)

    # Load BioBERT pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    biobert = AutoModel.from_pretrained('dmis-lab/biobert-v1.1').to(device)
    biobert.train()

    gnn_model = GNNEncoder(9, 256, 768, 'GAT').to(device)
    gnn_model.train()

    opt = torch.optim.AdamW([
                {'params': gnn_model.parameters(), 'lr': 1e-3},
                {'params': biobert.parameters(), 'lr': 1e-9}
            ])

    
    train(gnn_model, biobert, tokenizer, loader, device, 50)
    
    
    
