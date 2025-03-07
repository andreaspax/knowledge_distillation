import torch
import transformers
import data
import utils
import os
import models

from tqdm import tqdm

device = utils.get_device()


def train_kd(epochs=5):
    # 1. Load tokenizer with padding token properly configured
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B", 
        token=os.getenv("HF_TOKEN_CURSOR")
    )
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load teacher with CPU-friendly settings
    print("Loading teacher model...")
    teacher = transformers.AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B", 
        token=os.getenv("HF_TOKEN_CURSOR"),
        torch_dtype=torch.float16,  # Use FP16 for memory savings
        low_cpu_mem_usage=True
    ).to(device)  
    
    # 3. Create student model
    print("Creating student model...")
    student = models.Decoder(
        vocab_size=len(tokenizer),
        embed_size=512,
        num_layers=4,
        heads=8,
        max_seq_len=260
    ).to(device) 

    grad_accumulation_steps = 4  # Accumulate gradients
    
    train_loader = data.get_text8_dataloader(tokenizer, batch_size=16, seq_length=256)
    # Smaller test dataset to reduce memory
    test_loader = data.get_text8_dataloader(tokenizer, max_text_length=500000, batch_size=4)
    
    criterion = models.DistillationLoss(alpha=0.5, temperature=4, pad_token_id=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    teacher.eval()
    student.train()
    
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch
        
        for i, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # add BOS token to input
            input_ids = torch.cat([torch.ones(input_ids.shape[0], 1, dtype=torch.long, device=device) * tokenizer.bos_token_id, input_ids], dim=1)
            labels = torch.cat([labels, torch.ones(labels.shape[0], 1, dtype=torch.long, device=device) * tokenizer.eos_token_id], dim=1)
            
            # Forward pass through teacher (keeping on its device)
            with torch.no_grad():
                teacher_outputs = teacher(input_ids.to(teacher.device)).logits
                # Move teacher outputs to the same device as student
                teacher_outputs = teacher_outputs.to(device)
            
            # Forward pass through student
            student_outputs = student(input_ids)
            
            # Compute loss and scale it for gradient accumulation
            loss = criterion(student_outputs, teacher_outputs, labels) / grad_accumulation_steps
            
            # Backpropagation with gradient accumulation
            loss.backward()
            
            # Only step every grad_accumulation_steps batches
            if (i + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            # Update progress bar
            loop.set_postfix(loss=loss.item() * grad_accumulation_steps)
        
        # Make sure to step after the epoch if there are leftover batches
        if len(train_loader) % grad_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            
        scheduler.step()

        # Evaluate on a small subset to save memory
        print("Evaluating...")
        student_ppl = models.evaluate_perplexity(student, test_loader)
        
        # No need to evaluate teacher perplexity each time to save memory
        if epoch == 0 or epoch == epochs-1:
            teacher_ppl = models.evaluate_perplexity(teacher, test_loader)
            print(f"Teacher Perplexity: {teacher_ppl:.2f}, Student Perplexity: {student_ppl:.2f}")
        else:
            print(f"Student Perplexity: {student_ppl:.2f}")


if __name__ == "__main__":
    train_kd(epochs=5)