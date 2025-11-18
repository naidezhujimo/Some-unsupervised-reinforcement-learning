import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import random
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import json
from tqdm import tqdm
import time
import argparse
from collections import defaultdict, deque
import hashlib
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ICMConfig:
    alpha: float = 50.0  # 互预测性权重
    initial_temp: float = 10.0  # 模拟退火初始温度
    final_temp: float = 0.01  # 最低温度
    cooling_rate: float = 0.99  # 温度衰减率
    init_samples: int = 8  # 初始随机样本数
    max_iterations: int = 1000
    batch_size: int = 4
    max_context_examples: int = 5  # 最大上下文示例数
    context_window: int = 2048  # 模型上下文窗口
    fix_steps: int = 10  # 一致性修复迭代次数
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 620
    cache_size: int = 1000  # 概率缓存大小
    priority_weight: float = 100.0  # 相关样本采样权重

    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning('cuda不可用, 使用cpu')
            self.device = "cpu"

class ProbabilityCache:
    """缓存概率计算结果以提高效率"""
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = deque()

    def _get_key(self, x, labels, context):
        context_str = '|'.join([f"{d['x']}:{d['y']}" for d in context])
        labels_str = ','.join(sorted(labels))
        content = F"{x}|{labels_str}|{context_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, x, labels, context):
        key = self._get_key(x, labels, context)
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.appendleft(key)
            return self.cache[key]
        return None

    def set(self, x, labels, context, probs):
        key = self._get_key(x, labels, context)
        if len(self.cache) >= self.max_size:
            oldest_key = self.access_order.pop()
            del self.cache[oldest_key]
        self.cache[key] = probs
        self.access_order.appendleft(key)

class ConsistencyChecker(ABC):
    """逻辑一致性检查抽象基类"""
    @abstractmethod
    def check_pair(self, x_i, y_i, x_j, y_j):
        """检查两标签是否一致(True=一致, False=冲突)"""
        pass

    def get_inconsistent_pairs(self, data):
        inconsistent = []

        # 构建问题到索引的映射
        question_to_indices = defaultdict(list)
        for i, item in enumerate(data):
            q = self.extract_question(item['x'])
            question_to_indices[q].append(i)

        # 只检查同一问题的样本对
        for indices in question_to_indices.values():
            if len(indices) > 1:
                for i_idx, i in enumerate(indices):
                    for j in indices[i_idx+1:]:
                        if not self.check_pair(data[i]['x'], data[i]['y'], data[j]['x'], data[j]['y']):
                            inconsistent.append((i, j))
        
        return inconsistent
    
    @abstractmethod
    def get_consistent_labels(self, x_i, x_j):
        """返回所有逻辑可能的标签组合"""
        pass

    def extract_question(self, text):
        """提取问题部分"""
        if "Question:" in text:
            return text.split("Claim:")[0].strip()
        return text.split("\n")[0] if "\n" in text else text

class MathConsistencyChecker(ConsistencyChecker):
    """数学验证任务一致性检查"""
    def __init__(self):
        self.answer_patterns = ['The answer is', '答案是', 'Answer:']

    def extract_answer(self, text):
        for pattern in self.answer_patterns:
            if pattern in text:
                start = text.find(pattern) + len(pattern)
                answer_part = text[start:start+20]
                numbers = re.findall(r'-?\d+\.?\d*', answer_part)
                if numbers:
                    return numbers[0]
        return None
    
    def check_pair(self, x_i, y_i, x_j, y_j):
        q_i, q_j = self.extract_question(x_i), self.extract_question(x_j)
        if q_i != q_j:
            return True
        
        ans_i, ans_j = self.extract_answer(x_i), self.extract_answer(x_j)
        if ans_i and ans_j and ans_i != ans_j:
            if (y_i == 'True' and y_j == 'True') or (y_i == 'False' and y_j == 'False'):
                return False
        return True
    
    def get_consistent_labels(self, x_i, x_j):
        q_i, q_j = self.extract_question(x_i), self.extract_question(x_j)
        ans_i, ans_j = self.extract_answer(x_i), self.extract_answer(x_j)
        
        if q_i == q_j and ans_i and ans_j and ans_i != ans_j:
            # 同一问题不同答案：不能同时为True
            return [('True', 'False'), ('False', 'True'), ('False', 'False')]
        return [('True', 'True'), ('True', 'False'), ('False', 'True'), ('False', 'False')]

class AlpacaConsistencyChecker(ConsistencyChecker):
    def extract_responses(self, text):
        if 'Response A:' in text and 'Response B:' in text:
            parts = text.split('Response A:')[1].split('Response B:')
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
        return None, None
    
    def check_pair(self, x_i, y_i, x_j, y_j):
        resp_a_i, resp_b_i = self.extract_responses(x_i)
        resp_a_j, resp_b_j = self.extract_responses(x_j)

        if resp_a_i and resp_b_i and resp_a_j and resp_b_j:
            # 如果是相同的响应对但顺序相反
            if (resp_a_i == resp_b_j and resp_b_i == resp_a_j):
                if y_i == "True" and y_j == "True":
                    return False
        return True
    
    def get_consistent_labels(self, x_i, x_j):
        resp_a_i, resp_b_i = self.extract_responses(x_i)
        resp_a_j, resp_b_j = self.extract_responses(x_j)
        
        if (resp_a_i and resp_b_i and resp_a_j and resp_b_j and
            resp_a_i == resp_b_j and resp_b_i == resp_a_j):
            # 反对称情况
            return [('True', 'False'), ('False', 'True'), ('False', 'False')]
        return [('True', 'True'), ('True', 'False'), ('False', 'True'), ('False', 'False')]

class ICMCore:
    def __init__(self, model, tokenizer, config, consistency_checker):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.checker = consistency_checker
        self.cache = ProbabilityCache(config.cache_size)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.config.device)
        self.model.eval()

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        logger.info(f"ICMCore initialized on {config.device}")

    def build_prompt(self, x, context):
        parts = []
        for ex in context[-self.config.max_context_examples:]:
            parts.append(f"Input: {ex['x']}\nLabel: {ex['y']}")
        parts.append(f"Input: {x}\nLabel:")
        return "\n\n".join(parts)
    
    @torch.no_grad()
    def get_label_prob_batch(self, x_list, labels, context_list):
        """批量计算条件概率"""
        results = []

        for i, (x, context) in enumerate(zip(x_list, context_list)):
            cached = self.cache.get(x, labels, context)
            if cached is not None:
                results.append(cached)
                continue

            prompt = self.build_prompt(x, context)
            prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt')

            probs = {}
            for label in labels:
                full_text = prompt + ' ' + label
                inputs = self.tokenizer(full_text, return_tensors='pt', truncation=True, max_length=self.config.context_window, return_offsets_mapping=True).to(self.config.device)

                prompt_len = len(prompt_ids[0])
                label_start = None
                offset_mapping = inputs['offset_mapping'][0]

                for idx, (start, end) in enumerate(offset_mapping):
                    if start >= len(prompt) and label_start is None:
                        label_start = idx
                        break
                
                if label_start is None:
                    probs[label] = 1.0 / len(labels)  # 回退
                    continue

                outputs = self.model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
                logits = outputs.logits

                label_ids = self.tokenizer.encode(label, add_special_tokens=False)
                log_prob = 0.0
                valid_tokens = 0

                for j, token_id in enumerate(label_ids):
                    pos = label_start + j
                    if pos < logits.shape[1]:
                        token_log_prob = torch.log_softmax(logits[0, pos], dim=-1)[token_id].item()
                        log_prob += token_log_prob
                        valid_tokens += 1

                if valid_tokens > 0:
                    probs[label] = np.exp(log_prob / valid_tokens)  # 平均token概率
                else:
                    probs[label] = 1.0 / len(labels)

            # 归一化
            total = sum(probs.values())
            if total > 0:
                probs = {k: v/total for k, v in probs.items()}
            else:
                probs = {label: 1.0/len(labels) for label in labels}
            
            self.cache.set(x, labels, context, probs)
            results.append(probs)
        
        return results
    
    @torch.no_grad()
    def get_label_prob(self, x, labels, context):
        """单个样本的条件概率计算"""
        return self.get_label_prob_batch([x], labels, [context])[0]

    def compute_mutual_predictability(self, data):
        """计算互预测性 P_θ(D)"""
        if len(data) <= 1:
            return 0.0
        
        total_log_prob = 0.0
        batch_size = min(self.config.batch_size, len(data))

        for start_idx in tqdm(range(0, len(data), batch_size), desc='Mutual Pred', leave=False):
            end_idx = min(start_idx + batch_size, len(data))
            batch_data = data[start_idx: end_idx]

            x_list = [item['x'] for item in batch_data]
            context_list = []

            for i, item in enumerate(batch_data):
                context = [d for j, d in enumerate(data) if j != start_idx + i]
                context_list.append(context)

            all_labels = list(set(item['y'] for item in data))
            all_probs = self.get_label_prob_batch(x_list, all_labels, context_list)
            
            for i, probs in enumerate(all_probs):
                true_labels = batch_data[i]['y']
                prob = probs.get(true_labels, 1e-10)
                total_log_prob += np.log(prob)

        return total_log_prob / len(data)
    
    def compute_consistency_penalty(self, data):
        """计算一致性惩罚 I(D)"""
        return len(self.checker.get_inconsistent_pairs(data))

    def compute_score(self, data):
        """计算U(D) = α·P(D) - I(D)"""
        if not data:
            return -np.inf
            
        mutual = self.compute_mutual_predictability(data)
        penalty = self.compute_consistency_penalty(data)
        score = self.config.alpha * mutual - penalty

        return score

    def fix_inconsistencies(self, data):
        if len(data) <= 1:
            return data
        
        data = [item.copy() for item in data]
        improved = True
        steps = 0

        while improved and steps < self.config.fix_steps:
            improved = False
            pairs = self.checker.get_inconsistent_pairs(data)
            
            if not pairs:
                break

            # 随机采样一些不一致对进行修复
            sample_pairs = random.sample(pairs, min(10, len(pairs)))

            for i, j in sample_pairs:
                x_i, x_j = data[i]['x'], data[j]['x']
                label_options = self.checker.get_consistent_labels(x_i, x_j)

                best_score = -np.inf
                best_labels = None
                current_score = self.compute_score(data)

                orig_y_i, orig_y_j = data[i]['y'], data[j]['y']

                for y_i_new, y_j_new in label_options:
                    data[i]['y'], data[j]['y'] = y_i_new, y_j_new
                    score = self.compute_score(data)

                    if score > best_score:
                        best_score = score
                        best_labels = (y_i_new, y_j_new)
                
                if best_labels and best_score > current_score:
                    data[i]['y'], data[j]['y'] = best_labels
                    improved = True
                else:
                    data[i]['y'], data[j]['y'] = orig_y_i, orig_y_j
            
            steps += 1

        return data
    
    def select_example(self, unlabeled_indices, labeled, unlabeled_data):
        if not labeled or random.random() < 0.3:
            idx = random.choice(list(unlabeled_indices))
            return unlabeled_data[idx], idx
        
        # 基于问题相似度采样
        ref_item = random.choice(labeled)
        ref_question = self.checker.extract_question(ref_item['x'])
        
        scores = []
        candidates = []

        for idx in unlabeled_indices:
            question = self.checker.extract_question(unlabeled_data[idx])
            # 简单相似度：问题前缀匹配
            similarity = 1.0 if question == ref_question else 0.1
            scores.append(similarity)
            candidates.append(idx)

        # 加权采样
        if sum(scores) > 0:
            idx = random.choices(candidates, weights=scores, k=1)[0]
        else:
            idx = random.choice(candidates)
            
        return unlabeled_data[idx], idx
    
    def run(self, unlabeled, labels, max_iter=None):
        """通过模拟退火搜索优化标注, 最大化U(D)评分"""
        max_iter = max_iter or (min(len(unlabeled), self.config.max_iterations))

        unlabeled_indices = set(range(len(unlabeled)))
        labeled = []

        init_indices = random.sample(list(unlabeled_indices), min(self.config.init_samples, len(unlabeled)))

        for idx in init_indices:
            labeled.append({
                'x': unlabeled[idx], 
                'y': random.choice(labels), 
                'idx': idx
            })
            unlabeled_indices.remove(idx)

        # 初始一致性修复
        labeled = self.fix_inconsistencies(labeled)
        current_score = self.compute_score(labeled)
        
        logger.info(f"初始化完成: {len(labeled)} 个样本, 评分={current_score:.4f}")
        
        start_time = time.time()
        progress_bar = tqdm(range(max_iter), desc="ICM优化进度")
        
        for iteration in progress_bar:
            if not unlabeled_indices:
                break

            temp = max(self.config.final_temp, self.config.initial_temp * (self.config.cooling_rate ** iteration))

            # 选择新样本进行标注
            x_new, idx_new = self.select_example(unlabeled_indices, labeled, unlabeled)
            
            # 预测最可能的标签
            probs = self.get_label_prob(x_new, labels, labeled)
            y_new = max(probs, key=probs.get)
            
            # 创建包含新标注的临时数据集
            new_item = {'x': x_new, 'y': y_new, 'idx': idx_new}
            temp_labeled = labeled + [new_item]
            temp_labeled = self.fix_inconsistencies(temp_labeled)  # 修复可能的不一致
            new_score = self.compute_score(temp_labeled)
            
            delta = new_score - current_score  # 评分变化量

            # 模拟退火接受决策
            accept = False
            if delta > 0:
                accept = True
            else:
                accept_prob = np.exp(delta / temp) if temp > 0 else 0
                accept = random.random() < accept_prob
            
            if accept:
                # 接受新标注
                labeled = temp_labeled
                current_score = new_score
                unlabeled_indices.remove(idx_new)
                
                if iteration % 50 == 0:
                    logger.info(f"迭代 {iteration}: 评分={current_score:.4f}, "
                               f"变化量={delta:.4f}, 温度={temp:.4f}")
            
            progress_bar.set_postfix({
                'score': f'{current_score:.3f}',
                'labeled': len(labeled),
                'temp': f'{temp:.3f}'
            })
        
        elapsed = time.time() - start_time
        logger.info(f"优化完成: {len(labeled)} 个标注样本, 最终评分={current_score:.4f}, "
                   f"总耗时={elapsed:.2f}秒")
        
        return labeled
    
class ICMTrainer:
    def __init__(self, model_name, task, config):
        self.config = config or ICMConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        task_map = {
            "math": MathConsistencyChecker,
            "alpaca": AlpacaConsistencyChecker,
        }

        if task not in task_map:
            logger.warning(f"任务 {task} 不支持，使用数学检查器")
            task = "math"
        
        checker_class = task_map[task]
        self.checker = checker_class()
        self.icm = ICMCore(self.model, self.tokenizer, self.config, self.checker)
    
    def run_demo(self, n_samples=50):
        logger.info(f"生成演示数据 ({n_samples} 个样本)...")
        
        # 数学验证演示数据
        math_questions = [
            ("5+5=?", "10", ["8", "9", "10", "11"]),
            ("3×4=?", "12", ["10", "12", "14", "16"]),
            ("15÷3=?", "5", ["3", "5", "6", "7"]),
            ("2²+3²=?", "13", ["10", "13", "25", "36"]),
            ("√16=?", "4", ["2", "4", "8", "16"]),
        ]
        
        demo_data = []
        for i in range(n_samples):
            # 随机选择问题和答案
            q, correct, wrongs = random.choice(math_questions)
            # 70%正确答案，30%错误答案
            answer = correct if random.random() > 0.3 else random.choice(wrongs)
            is_correct = answer == correct
            
            text = f"Question: {q}\nClaim: The answer is {answer}. This is {'correct' if is_correct else 'incorrect'}."
            demo_data.append(text)
        
        logger.info("开始ICM优化...")
        results = self.icm.run(demo_data, ["True", "False"])
        
        self._analyze_results(results, demo_data)
        return results
    
    def _analyze_results(self, results, original_data):
        print("\n" + "="*80)
        print("ICM 演示结果分析")
        print("="*80)
        
        # 统计标签分布
        label_counts = defaultdict(int)
        for item in results:
            label_counts[item['y']] += 1
        
        print(f"总样本数: {len(results)}")
        for label, count in label_counts.items():
            percentage = count/len(results)*100
            print(f"标签 '{label}': {count} 样本 ({percentage:.1f}%)")
        
        # 显示前10个样本的标注结果
        print("\n前10个样本标注详情:")
        print("-" * 40)
        for i, item in enumerate(results[:10]):
            x_short = item['x'].replace('\n', ' ')[:80]
            print(f"{i+1:2d}. {x_short}... → {item['y']}")
        
        # 检查最终一致性
        inconsistent_pairs = self.checker.get_inconsistent_pairs(results)
        print(f"\n逻辑不一致对数量: {len(inconsistent_pairs)}")
        
        # 计算最终评分
        final_score = self.icm.compute_score(results)
        print(f"最终评分 U(D): {final_score:.4f}")

    def save_results(self, results, path):
        output = {
            'metadata': {
                'timestamp': time.time(),
                'config': {k: v for k, v in self.config.__dict__.items() 
                          if not k.startswith('_')},
                'total_samples': len(results)
            },
            'results': results
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"结果保存至: {path}")

def main():
    parser = argparse.ArgumentParser(description="ICM无监督引出算法")
    parser.add_argument("--model", default="gpt2",
                       help="预训练模型名称或路径")
    parser.add_argument("--task", default="math", choices=["math", "alpaca"],
                       help="任务类型: 数学验证或偏好比较")
    parser.add_argument("--n_samples", type=int, default=50,
                       help="演示样本数量")
    parser.add_argument("--alpha", type=float, default=50.0,
                       help="互预测性权重系数")
    parser.add_argument("--init_samples", type=int, default=8,
                       help="初始随机标注样本数量")
    parser.add_argument("--max_iter", type=int, default=500,
                       help="最大迭代次数")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--output", default="icm_results.json",
                       help="输出文件路径")
    
    args = parser.parse_args()
    
    config = ICMConfig(
        alpha=args.alpha,
        init_samples=args.init_samples,
        max_iterations=args.max_iter,
        seed=args.seed
    )
    
    try:
        trainer = ICMTrainer(args.model, args.task, config)
        results = trainer.run_demo(args.n_samples)
        trainer.save_results(results, args.output)
        
        logger.info("ICM演示完成!")
        
    except Exception as e:
        logger.error(f"运行失败: {e}")
        raise

if __name__ == "__main__":
    main()