from transformers import AutoTokenizer, AutoModel
from torch_geometric.loader import DataLoader
from gnn_model import GNNEncoder
from data import CustomDataset
from tqdm import tqdm
import torch
import wandb

from validation import validate

wandb.init(project="train_langval")

def train(graph_model, text_model, tokenizer, loader, device, epochs):
    count = 0
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            count = count  + 1
            opt.zero_grad()

            # Convert text to tensors
            tokens = tokenizer.batch_encode_plus(batch.y, add_special_tokens=True, padding='max_length', truncation=True, max_length=77 ,return_tensors="pt", return_attention_mask=True)
            attention_mask = tokens.attention_mask.to(device)
            tokens = tokens["input_ids"].to(device)
            with torch.no_grad():
                text_features = text_model(tokens)
                last_hidden_state = text_features.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
            
            graph_features = graph_model(batch)
            labels = torch.arange(batch.batch.max()+1 ,dtype=torch.long, device=device)

            loss = gnn_model.loss(graph_features, mean_embeddings, labels)
            loss.backward()
            opt.step()
            total_loss = total_loss + loss
        
        wandb.log({'loss': total_loss/len(loader)})
        if epoch > 99 and epoch % 100 == 0:
            graph_model.eval()
            res = validate(graph_model, text_model, tokenizer, device)
            wandb.log({'halicin': res[0]})
            wandb.log({'overlap': res[1]})
            graph_model.train()
            torch.save(gnn_model.state_dict(), 'gnn' + str(epoch) + '.pth')
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load data
    dataset = CustomDataset('./parse_data/sum_smiles.csv')
    loader = DataLoader(dataset, batch_size=36, shuffle=True)

    # Load BioBERT pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    biobert = AutoModel.from_pretrained('dmis-lab/biobert-v1.1').to(device)
    biobert.eval()

    gnn_model = GNNEncoder(9, 256, 768, 'GAT').to(device)
    gnn_model.train()

    opt = torch.optim.AdamW(gnn_model.parameters(), 1e-4)

    
    train(gnn_model, biobert, tokenizer, loader, device, 5000)
    
    
    
