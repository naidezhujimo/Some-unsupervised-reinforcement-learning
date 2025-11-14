import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

class EntropyMinimization:
    def __init__(self, model_name='gpt2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def token_level_entropy(self, logits):
        """计算标记级熵"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy
    
    def sequence_level_entropy(self, logits, attention_mask):
        """计算序列级熵"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 计算每个位置的熵
        token_entropies = -torch.sum(probs * log_probs, dim=-1)

        # 对有效标记的熵求平均
        valid_tokens = attention_mask.sum(dim=1)
        sequence_entropies = (token_entropies * attention_mask).sum(dim=1) / valid_tokens

        return sequence_entropies
    
class EMFTTrainer(EntropyMinimization):
    """EM-FT: 基于标记级熵的微调"""
    def __init__(self, model_name='gpt2',lr=1e-5):
        super().__init__(model_name)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def generate_training_data(self, prompts, num_samples=4):
        self.model.eval()
        training_data = {'input_ids': [], 'attention_mask': []}

        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                for _ in range(num_samples):
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 100,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                    training_data['input_ids'].append(outputs.cpu())
                    training_data['attention_mask'].append(torch.ones_like(outputs).cpu())

        return training_data

    def compute_entropy_loss(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 计算每个位置的熵
        token_entropies = self.token_level_entropy(logits)

        # 只计算非填充位置的熵
        valid_entropies = token_entropies[attention_mask.bool()]

        # 返回平均熵作为损失
        return valid_entropies.mean()

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.compute_entropy_loss(batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, prompts, num_epochs=3, batch_size=2):
        print("生成训练数据...")
        training_data = self.generate_training_data(prompts)
        
        input_ids = torch.cat(training_data[input_ids], dim=0)
        attention_mask = torch.cat(training_data['attention_mask'], dim=0)

        num_batches = math.ceil(input_ids.shape[0] / batch_size)

        print(f"开始训练，总批次数: {num_batches}")

        for epoch in range(num_epochs):
            total_loss = 0
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, input_ids.shape[0])

                batch = {
                    "input_ids": input_ids[start_idx:end_idx],
                    "attention_mask": attention_mask[start_idx:end_idx]
                }

                loss = self.train_step(batch)
                total_loss += loss

                if i % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {i+1}/{num_batches}, Loss: {loss:.4f}")
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")

class EMRLLoss(EntropyMinimization):
    """EM-RL: 基于强化学习的熵最小化损失函数"""
    def __init__(self, model_name='gpt2', beta=0.001):
        super().__init__(model_name)
        self.beta = beta

    def compute_reward(self, logits, attention_mask, reward_type='token'):
        if reward_type == 'token':
            token_entropies = self.token_level_entropy(logits)
            valid_entropies = token_entropies[attention_mask.bool()]
            reward = -valid_entropies.mean()
        else:
            sequence_entropies = self.sequence_level_entropy(logits, attention_mask)
            reward = -sequence_entropies.mean()
        return reward
    
    def compute_kl_penalty(self, logits, ref_logits, attention_mask):
        log_probs = F.log_softmax(logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        
        kl_div = F.kl_div(
            input=log_probs,
            target=ref_probs,
            reduction='none'
        ).sum(dim=-1)
        
        valid_kl = kl_div[attention_mask.bool()]
        return valid_kl.mean()

    def compute_loss(self, logits, ref_logits, attention_mask, reward_type='token'):
        # 计算奖励（负熵）
        reward = self.compute_reward(logits, attention_mask, reward_type)

        # 计算KL惩罚
        kl_penalty = self.compute_kl_penalty(logits, ref_logits, attention_mask)

        # 总损失 = -奖励 + β * KL惩罚
        total_loss = -reward + self.beta * kl_penalty

        return total_loss, reward.item(), kl_penalty.item()

class EMINFInference(EntropyMinimization):
    """EM-INF: 推理时logits优化"""
    def __init__(self, model_name='gpt2'):
        super().__init__(model_name)
    
    def optimize_logits(self, logits, delta=0.3, num_steps=15, lr=0.1):
        """优化logits以减少熵"""
        # 将logits设为需要梯度
        optimized_logits = logits.clone().to(self.device).requires_grad_(True)
        optimizer = optim.Adam([optimized_logits], lr=lr)

        for step in range(num_steps):
            optimizer.zero_grad()

            # 计算当前熵
            current_entropy = self.token_level_entropy(optimized_logits.unsqueeze(0)).mean()

            # 损失函数: 超过阈值时最小化熵
            if current_entropy > delta:
                loss = current_entropy
                loss.backward()
                optimizer.step()
            else:
                break

        return optimized_logits.detach()

    def adaptive_temperature_scaling(self, logits, alpha=0.5, delta=0.3):
        """自适应温度缩放方法(对比基线)"""
        original_entropy = self.token_level_entropy(logits.unsqueeze(0)).mean()
        target_entropy = max(alpha * original_entropy, delta)

        # 二分查找最佳温度
        low_temp, high_temp = 0.1, 10.0
        tolerance = 1e-4
        max_iters = 20

        for _ in range(max_iters):
            mid_temp = (low_temp + high_temp) / 2
            scaled_logits = logits / mid_temp
            current_entropy = self.token_level_entropy(scaled_logits.unsqueeze(0)).mean()
            
            if abs(current_entropy - target_entropy) < tolerance:
                break
            
            if current_entropy > target_entropy:
                low_temp = mid_temp
            else:
                high_temp = mid_temp
        
        return logits / mid_temp

    def generate_with_em_inf(self, prompt, method='logit_optimization', max_length=100, **kwargs):
        """使用EM-INF生成文本"""
        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        generated_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length):
                with torch.no_grad():
                    outputs = self.model(input_ids=generated_ids, attention_mask=attention_mask)
                    next_token_logits = outputs.logits[:, -1, :]
                
                if method == 'logit_optimization':
                    # EM-INF: logit优化
                    with torch.enable_grad():
                        optimized_logits = self.optimize_logits(
                            next_token_logits, 
                            delta=kwargs.get('delta', 0.3),
                            num_steps=kwargs.get('num_steps', 15)
                        )
                elif method == "adaptive_temp":
                    # 自适应温度缩放（基线）
                    optimized_logits = self.adaptive_temperature_scaling(
                        next_token_logits,
                        alpha=kwargs.get('alpha', 0.5),
                        delta=kwargs.get('delta', 0.3)
                    )
                else:
                    optimized_logits = next_token_logits
                
                probs = F.softmax(optimized_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones(1, 1, device=self.device)
                ], dim=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def demo_em_ft():
    """演示EM-FT"""
    print("=== EM-FT 演示 ===")
    
    prompts = [
        "Solve the equation: 2x + 5 = 13",
        "Calculate the area of a circle with radius 5",
        "What is the derivative of x^2?",
    ]
    
    trainer = EMFTTrainer("gpt2", learning_rate=1e-5)
    
    # 训练模型
    trainer.train(prompts, num_epochs=2, batch_size=2)
    
    # 测试生成
    test_prompt = "Solve the equation: 3x - 7 = 14"
    inputs = trainer.tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = trainer.model.generate(
            **inputs,
            max_length=50,
            do_sample=True,
            temperature=0.7
        )
    
    result = trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"生成结果: {result}")

def demo_em_inf():
    """演示EM-INF"""
    print("\n=== EM-INF 演示 ===")
    
    generator = EMINFInference("gpt2")
    
    test_prompts = [
        "Explain quantum physics in simple terms:",
        "Write a Python function to calculate fibonacci numbers:",
    ]
    
    methods = [
        ("贪婪解码", None),
        ("自适应温度缩放", "adaptive_temp"),
        ("EM-INF Logit优化", "logit_optimization")
    ]
    
    for prompt in test_prompts:
        print(f"\n输入: {prompt}")
        
        for method_name, method in methods:
            if method is None:
                # 基线：贪婪解码
                result = generator.generate_with_em_inf(
                    prompt, 
                    method="greedy",
                    max_length=50
                )
            else:
                result = generator.generate_with_em_inf(
                    prompt, 
                    method=method,
                    max_length=50
                )
            
            print(f"{method_name}: {result}\n")

def analyze_entropy_reduction():
    """分析熵减少效果"""
    print("\n=== 熵减少分析 ===")
    
    analyzer = EntropyMinimization("gpt2")
    
    test_text = "The quick brown fox jumps over the lazy dog"
    inputs = analyzer.tokenizer(test_text, return_tensors="pt")
    inputs = {k: v.to(analyzer.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = analyzer.model(**inputs)
        logits = outputs.logits
    
    # 原始熵
    original_entropy = analyzer.token_level_entropy(logits)
    print(f"原始标记级熵: {original_entropy.mean().item():.4f}")
    
    # 应用EM-INF后的熵
    em_inf = EMINFInference("gpt2")
    optimized_logits = em_inf.optimize_logits(logits[0, -1, :])
    optimized_entropy = em_inf.token_level_entropy(optimized_logits.unsqueeze(0))
    print(f"优化后标记级熵: {optimized_entropy.mean().item():.4f}")
    
    entropy_reduction = (original_entropy.mean() - optimized_entropy.mean()).item()
    print(f"熵减少量: {entropy_reduction:.4f}")

if __name__ == "__main__":
    # 运行演示
    demo_em_inf()
    analyze_entropy_reduction()
    
    # 注意：EM-FT训练需要较长时间，取消注释以运行
    # demo_em_ft()