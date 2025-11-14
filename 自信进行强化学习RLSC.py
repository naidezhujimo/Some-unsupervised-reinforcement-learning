import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import numpy as np
import json
from tqdm import tqdm

class RLSCConfig:
    def __init__(self):
        self.model_name = 'gpt2'
        self.num_samples_per_question = 16
        self.temperature = 0.5
        self.lr = 1e-5
        self.weight_decay = 0.01
        self.num_train_steps = 20
        self.batch_size = 1
        self.max_length = 1024
        self.alpha = 0.1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 42

class MathDataset(Dataset):
    def __init__(self, questions, tokenizer, max_length=1024):
        self.questions = questions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        encoding = self.tokenizer(
            question,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "question": question
        }
    
class RLSCLoss:
    @staticmethod
    def compute_sequence_log_probabilities(model, input_ids, attention_mask, temperature=1.0):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 计算每个token的对数概率
        shift_logits = logits[:, :-1, :]  # 移除最后一个token的预测
        shift_labels = input_ids[:, 1:]  # 移除第一个token的标签

        shift_logits = shift_logits / temperature

        # 计算对数概率
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # 获取目标token的对数概率
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        shift_attention_mask = attention_mask[:, 1:]
        masked_token_log_probs = token_log_probs * shift_attention_mask

        # 序列总对数概率
        sequence_log_probs = masked_token_log_probs.sum(dim=-1)

        return sequence_log_probs
    
    @staticmethod
    def compute_old_probabilities(old_model, completions, tokenizer, temperature=1.0):
        if not completions:
            return torch.tensor([])
        
        encodings = tokenizer(
            completions,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(old_model.device)
        attention_mask = encodings['attention_mask'].to(old_model.device)

        # 计算对数概率
        with torch.no_grad():
            log_probs = RLSCLoss.compute_sequence_log_probabilities(
                old_model, input_ids, attention_mask, temperature
            )

        probs = F.softmax(log_probs, dim=0)

        return probs

    @staticmethod
    def compute_rlsc_loss(current_model, old_model, completions, tokenizer, alpha):
        if not completions:
            return torch.tensor(0.0), {}

        # 使用旧模型计算概率
        with torch.no_grad():
            old_probs = RLSCLoss.compute_old_probabilities(old_model, completions, tokenizer, temperature=1.0)

        # 使用当前模型计算对数概率
        encodings = tokenizer(completions, padding=True, truncation=True, max_length=1024, return_tensors='pt')

        input_ids = encodings['input_ids'].to(current_model.device)
        attention_mask = encodings['attention_mask'].to(current_model.device)

        current_log_probs = RLSCLoss.compute_sequence_log_probabilities(
            current_model, input_ids, attention_mask, temperature=1.0
        )

        # 计算基础损失 L1
        loss_l1 = -torch.sum(old_probs * current_log_probs)

        # 计算平滑损失 L2
        weights = old_probs + alpha
        normalized_weights = weights / weights.sum()
        loss_l2 = -torch.sum(normalized_weights * current_log_probs)

        loss = loss_l2

        metrics = {
            "loss": loss.item(),
            "loss_l1": loss_l1.item(),
            "loss_l2": loss_l2.item(),
            "avg_old_prob": old_probs.mean().item(),
            "avg_current_log_prob": current_log_probs.mean().item(),
            "confidence_increase": (current_log_probs.exp().mean() - old_probs.mean()).item()
        }
        
        return loss, metrics

class RLSCGenerator:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def generate_completions(self, question):
        input_encoding = self.tokenizer(question, return_tensors='pt')
        input_ids = input_encoding['input_ids'].to(self.config.device)
        attention_mask = input_encoding['attention_mask'].to(self.config.device)

        completions = []

        for _ in range(self.config.num_samples_per_question):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=min(input_ids.shape[1] + 200, self.config.max_length),
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
        
            completion = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

            completions.append(completion)

        return completions

class RLSCTrainer:
    def __init__(self, config):
        self.config = config
        self.setup_models()
        self.setup_datas()
    
    def setup_models(self):
        print("Loading model and tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.current_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=torch.float32
        ).to(self.config.device)
        
        self.old_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=torch.float32
        ).to(self.config.device)
        
        for param in self.old_model.parameters():
            param.requires_grad = False

        self.current_model.train()
        self.old_model.eval()

        print(f"Model loaded on {self.config.device}")

    def setup_datas(self):
        self.train_questions = [
            "Solve for x: 2x + 5 = 13",
            "What is the area of a circle with radius 5?",
            "Find the derivative of f(x) = 3x^2 + 2x - 1",
            "Calculate the distance between points (2, 3) and (5, 7)",
            "Simplify: (2x^2 + 3x - 1) + (x^2 - 2x + 4)",
            "Solve the equation: x^2 - 5x + 6 = 0",
            "What is the value of sin(π/2)?",
            "Find the integral of ∫(2x + 3) dx",
            "Calculate 15% of 200",
            "Solve the system: 2x + y = 7, x - y = 1"
        ]
        
        self.dataset = MathDataset(self.train_questions, self.tokenizer, self.config.max_length)
        self.dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        print(f"Loaded {len(self.train_questions)} training questions")
    
    def setup_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.current_model.named_parameters()
                if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.current_model.named_parameters()
                if any (nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.lr)

        print('Optimizer setup complete')

    def train_epoch(self, epoch):
        self.current_model.train()

        total_loss = 0
        total_confidence_increase = 0
        num_batches = len(self.dataloader)

        progress_bar = tqdm(self.dataloader, desc= f'Epoch {epoch}')

        for batch_idx, batch in enumerate(progress_bar):
            question = batch['question'][0]  # 因为batch_size=1

            generator = RLSCGenerator(self.old_model, self.tokenizer, self.config)
            completions = generator.generate_completions(question)

            if not completions:
                continue

            loss, metrics = RLSCLoss.compute_rlsc_loss(
                self.current_model,
                self.old_model,
                completions,
                self.tokenizer,
                alpha=self.config.alpha
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += metrics['loss']
            total_confidence_increase += metrics["confidence_increase"]

            progress_bar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "conf_inc": f"{metrics['confidence_increase']:.4f}"
            })

            if batch_idx % 5 == 0:
                self.update_old_model()

        avg_loss = total_loss / num_batches
        avg_confidence_increase = total_confidence_increase / num_batches

        return {
            "avg_loss": avg_loss,
            "avg_confidence_increase": avg_confidence_increase
        }
    
    def update_old_model(self):
        old_state_dict = self.old_model.state_dict()
        current_state_dict = self.current_model.state_dict()

        for key in current_state_dict:
            if key in old_state_dict:
                old_state_dict[key].copy_(current_state_dict[key])

    def train(self):
        print("Starting RLSC training...")
        print(f"Training for {self.config.num_train_steps} steps")
        print(f"Using alpha = {self.config.alpha}")
        
        self.setup_optimizer()

        training_history = []
        
        for step in range(self.config.num_train_steps):
            print(f'\n--- Training Step {step + 1}/{self.config.num_train_steps} ---')

            metrics = self.train_epoch(step + 1)
            training_history.append(metrics)

            print(f'Step {step + 1} Metrics: {metrics}')

            if (step + 1) % 5 == 0:
                self.save_checkpoint(step + 1)

        print("\nTraining completed!")
        return training_history

    def save_checkpoint(self, step: int):
        checkpoint = {
            "step": step,
            "current_model_state_dict": self.current_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__
        }
        
        filename = f"rlsc_checkpoint_step_{step}.pt"
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

    
    def evaluate_model(self, test_questions):
        self.current_model.eval()

        correct_answers = 0
        total_questions = len(test_questions)

        print(f"\nEvaluating on {total_questions} test questions...")
        
        for question in test_questions:
            generator = RLSCGenerator(self.current_model, self.tokenizer, self.config)
            completions = generator.generate_completions(question)

            if completions:
                with torch.no_grad():
                    old_probs = RLSCLoss.compute_old_probabilities(
                        self.old_model, completions, self.tokenizer
                    )

                best_idx = torch.argmax(old_probs).item()
                best_answer = completions[best_idx]

                correct_answer += 1

                print(f"Q: {question}")
                print(f"A: {best_answer}")
                print(f"Confidence: {old_probs[best_idx]:.4f}")
                print("-" * 50)
        
        accuracy = correct_answers / total_questions

        return {
            "accuracy": accuracy,
            "correct_answers": correct_answers,
            "total_questions": total_questions
        }

def main():
    config = RLSCConfig()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    trainer = RLSCTrainer(config)

    training_history = trainer.train()

    test_questions = [
        "What is 15% of 200?",
        "Solve for x: 3x - 7 = 8",
        "Calculate the area of a triangle with base 6 and height 4"
    ]

    eval_results = trainer.evaluate_model(test_questions)
    print(f"\nEvaluation Results: {eval_results}")
    
    trainer.current_model.save_pretrained("rlsc_final_model")
    trainer.tokenizer.save_pretrained("rlsc_final_model")
    print("Final model saved!")

if __name__ == "__main__":
    main()