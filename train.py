import torch
import transformers
import data
import utils
import os
import models

from tqdm import tqdm

device = utils.get_device()


def train_kd(epochs=5):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B", 
        token=os.getenv("HF_TOKEN_CURSOR")
    )

    teacher = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", 
    token=os.getenv("HF_TOKEN_CURSOR")
)
    
    teacher.to(device)

    # Create model
    student = models.Decoder(
            vocab_size=tokenizer.vocab_size,
            embed_size=512,  # Smaller for faster training
            num_layers=4,    # Fewer layers for faster training
            heads=8,
            max_seq_len=260
        )
    student.to(device)

    train_loader = data.get_text8_dataloader(tokenizer, batch_size=16)
    test_loader = data.get_text8_dataloader(tokenizer, max_text_length=1000000, batch_size=8)
    criterion = models.DistillationLoss(alpha=0.5, temperature=4, pad_token_id=tokenizer.pad_token_id)

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    teacher.eval()
    student.train()
    
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass through teacher
            with torch.no_grad():
                teacher_outputs = teacher(input_ids).logits
            
            # Forward pass through student
            student_outputs = student(input_ids)
            
            # Compute loss
            loss = criterion(student_outputs, teacher_outputs, labels)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())
        scheduler.step()

        student_ppl = models.evaluate_perplexity(student, test_loader)
        teacher_ppl = models.evaluate_perplexity(teacher, test_loader)
        print(f"Teacher Perplexity: {teacher_ppl:.2f}, Student Perplexity: {student_ppl:.2f}")


if __name__ == "__main__":
    train_kd(epochs=5)