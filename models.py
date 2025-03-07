import torch
import dotenv
from tqdm import tqdm
import utils

dotenv.load_dotenv()

device = utils.get_device()


class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, max_seq_len=1024):
        super().__init__()
        self.token_embed = torch.nn.Embedding(vocab_size, embed_size)
        
        # Position encoding
        self.pos_encoder = torch.nn.Parameter(torch.randn(1, max_seq_len, embed_size))
        
        # Use transformer encoder layers with causal attention mask
        # This provides the same functionality as a decoder without cross-attention
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=heads,
            dim_feedforward=embed_size * 4,
            activation="gelu",
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        # Token embeddings
        x = self.token_embed(x)  # (batch, seq_len, d_model)
        
        # Add positional encodings
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Create causal mask (so tokens can only attend to previous tokens)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        
        # Pass through transformer with causal mask
        x = self.transformer(x, mask=mask)
        logits = self.fc(x)
        return logits
    
class DistillationLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, temperature=4, pad_token_id=None):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    def forward(self, student_logits, teacher_logits, labels):
        # Soften distributions with temperature
        student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL Divergence loss (soft targets)
        soft_loss = self.kl_div(
            student_log_probs.view(-1, student_log_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1))
        ) * (self.temperature ** 2)
        
        # Cross-Entropy loss (hard targets)
        hard_loss = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

def evaluate_perplexity(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return torch.exp(torch.tensor(avg_loss)).item()  # Perplexity