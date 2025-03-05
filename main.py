import torch

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # Separate projections for clarity
        self.query =  torch.nn.Linear(embed_size, embed_size)
        self.key =  torch.nn.Linear(embed_size, embed_size)
        self.value =  torch.nn.Linear(embed_size, embed_size)
        self.out =  torch.nn.Linear(embed_size, embed_size)
        self.dropout =  torch.nn.Dropout(dropout)
        
    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape
        q = self.query(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        
        # Handle KV cache for inference
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
            
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)
        
        # Save current key and value states for next step
        present_kv = (k, v) if use_cache else None
        
        return (self.dropout(self.out(out)), present_kv) if use_cache else self.dropout(self.out(out))

class MLP( torch.nn.Module):
    def __init__(self, embed_size, expansion=4, dropout=0.1):
        super().__init__()
        self.net =  torch.nn.Sequential(
             torch.nn.Linear(embed_size, embed_size * expansion),
             torch.nn.GELU(),
             torch.nn.Dropout(dropout),
             torch.nn.Linear(embed_size * expansion, embed_size),
             torch.nn.Dropout(dropout)
        ) 
    def forward(self, x):
        return self.net(x)

class DecoderLayer( torch.nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super().__init__()
        self.norm1 =  torch.nn.LayerNorm(embed_size)
        self.attn = MultiHeadAttention(embed_size, heads, dropout)
        self.norm2 =  torch.nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, dropout=dropout)
        self.dropout1 =  torch.nn.Dropout(dropout)
        self.dropout2 =  torch.nn.Dropout(dropout)

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        # Layer norm and attention with caching
        residual = x
        x = self.norm1(x)
        
        # Modified attention call
        if use_cache:
            x, present_kv = self.attn(x, mask, past_kv, use_cache)
        else:
            x = self.attn(x, mask, past_kv, use_cache)
            
        x = residual + self.dropout1(x)
        
        # Layer norm and MLP
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.mlp(x))
        
        if use_cache:
            return x, present_kv
        return x

class DecoderConfig:
    def __init__(
        self,
        vocab_size=32000,
        embed_size=512,
        num_layers=6,
        heads=8,
        dropout=0.1,
        max_seq_len=1024,
        expansion_factor=4
    ):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.expansion_factor = expansion_factor

class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.token_embed = torch.nn.Embedding(vocab_size, embed_size)
        
        # Add a position encoder (simple linear projection of one-hot position vectors)
        self.pos_encoder = torch.nn.Linear(max_seq_len, embed_size)
        self.max_seq_len = max_seq_len
        
        self.dropout = torch.nn.Dropout(dropout)
        self.layers = torch.nn.ModuleList([
            DecoderLayer(embed_size, heads, dropout) for _ in range(num_layers)
        ])
        self.norm = torch.nn.LayerNorm(embed_size)
        self.head = torch.nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Get token embeddings
        token_emb = self.token_embed(x)
        
        # Create one-hot position encodings
        positions = torch.arange(seq_len, device=x.device)
        pos_one_hot = torch.nn.functional.one_hot(positions, num_classes=self.max_seq_len).float()
        pos_emb = self.pos_encoder(pos_one_hot)
        
        # Add to token embeddings
        x = token_emb + pos_emb.unsqueeze(0)
        x = self.dropout(x)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.head(self.norm(x)) 
    

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 32000  # typical tokenizer vocabulary size
    embed_size = 512
    num_layers = 6
    heads = 8

    # tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    # model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

    decoder = Decoder(vocab_size, embed_size, num_layers, heads)
    input_ids = torch.randint(0, vocab_size, (1, 128))  # example input
    output = decoder(input_ids)
    print(output.shape)  # should be [1, 128, vocab_size]