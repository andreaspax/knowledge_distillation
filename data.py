import torch
import requests
import os


def get_text8():
    # Check if text8 file exists and download it if not
    if not os.path.exists('text8'):
        print("Downloading text8 dataset...")
        r = requests.get("https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8")
        with open("text8", "wb") as f: f.write(r.content)
        
    # Open text8 file
    with open("text8", "r") as f:
        text8 = f.read()

    return text8

class Text8Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, seq_length=128, stride=64, max_length=None):
        """
        Create chunked dataset with sliding window from text8
        
        Args:
            tokenizer: HuggingFace tokenizer
            seq_length: Length of each sequence
            stride: Step size for the sliding window
            max_length: Optional maximum text length to use (for testing)
        """
        # Get the raw text
        text = get_text8()
        if max_length:
            text = text[:max_length]
        
        print(f"Tokenizing text (length: {len(text)})...")
        
        # Process in chunks to avoid memory issues
        chunk_size = 1000000 # 1M chars per chunk
        token_chunks = []
        
        # Tokenize text in chunks
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            tokens = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).input_ids[0]
            token_chunks.append(tokens)
            print(f"Processed chunk {i//chunk_size + 1}/{(len(text)-1)//chunk_size + 1}")
        
        # Concatenate all tokens
        all_tokens = torch.cat(token_chunks)
        print(f"Total tokens: {len(all_tokens)}")
        
        # Create sliding windows indices
        self.seq_length = seq_length
        self.examples = []
        
        for start_idx in range(0, len(all_tokens) - seq_length + 1, stride):
            self.examples.append(all_tokens[start_idx:start_idx + seq_length])
        
        print(f"Created {len(self.examples)} sequences with sliding window")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Get the sequence
        input_ids = self.examples[idx]
        
        # For causal language modeling, input and labels are the same
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone()
        }

def get_text8_dataloader(tokenizer, batch_size=8, seq_length=256, stride=128, max_text_length=None):
    """
    Create a dataloader for text8 with sliding window
    """
    dataset = Text8Dataset(
        tokenizer=tokenizer,
        seq_length=seq_length,
        stride=stride,
        max_length=max_text_length
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return dataloader