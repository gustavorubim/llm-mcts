import math
import ollama
from typing import List, Dict, Tuple, Optional
from transformers import GPT2Tokenizer
import re
import random

# Load the GPT-2 tokenizer (ideally, match this to your model's tokenizer)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

### Helper Classes and Functions

class Node:
    """Represents a node in the MCTS tree."""
    def __init__(self, sequence: List[int], log_prob_to_node: float = 0.0, parent: Optional['Node'] = None):
        self.sequence = sequence  # Token IDs
        self.log_prob_to_node = log_prob_to_node  # Log probability up to this node
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.total_quality = 0.0  # Sum of simulation qualities

def tokenize(text: str) -> List[int]:
    """Convert text to token IDs."""
    return tokenizer.encode(text)

def detokenize(token_ids: List[int]) -> str:
    """Convert token IDs back to text."""
    return tokenizer.decode(token_ids)

def get_top_k_tokens(sequence: List[int], k: int) -> List[Tuple[int, float]]:
    """Get the top-k next tokens and their log probabilities using Ollama."""
    prompt = detokenize(sequence)
    try:
        response = ollama.generate(
            model='deepseek-r1:14b',
            prompt=prompt,
            options={
                'num_predict': 1,
                'top_k': k,
                'logprobs': True
            }
        )
        logprobs = response.get('logprobs', [])
        if not logprobs:
            return []
        sorted_logprobs = sorted(logprobs, key=lambda x: x['logprob'], reverse=True)[:k]
        return [(item['token'], item['logprob']) for item in sorted_logprobs]
    except Exception:
        return []

def generate_completion(sequence: List[int], max_length: int) -> Tuple[List[int], float]:
    """Generate a completion up to max_length using the raw model."""
    prompt = detokenize(sequence)
    num_to_generate = max_length - len(sequence)
    if num_to_generate <= 0:
        return sequence, 0.0
    try:
        response = ollama.generate(
            model='deepseek-r1:14b',
            prompt=prompt,
            options={
                'num_predict': num_to_generate,
                'temperature': 0.0  # Greedy decoding
            }
        )
        generated_text = response.get('response', '')
        generated_tokens = tokenize(generated_text)
        completed_sequence = sequence + generated_tokens
        logprobs = response.get('logprobs', [])
        total_log_prob = sum(logprobs) if logprobs else 0.0
        return completed_sequence, total_log_prob
    except Exception:
        return sequence, 0.0

### MCTS Implementation

def ucb(node: Node, C: float = 1.0) -> float:
    """Calculate the UCB1 score for a node."""
    if node.visits == 0:
        return float('inf')
    parent_visits = node.parent.visits if node.parent else 1
    return (node.total_quality / node.visits) + C * math.sqrt(math.log(parent_visits) / node.visits)

def select(root: Node) -> Node:
    """Select a leaf node using UCB."""
    current = root
    while current.children:
        current = max(current.children, key=lambda n: ucb(n))
    return current

def expand(node: Node, k: int) -> Node:
    """Expand a node with top-k next tokens."""
    top_k = get_top_k_tokens(node.sequence, k)
    for token_id, log_prob in top_k:
        new_sequence = node.sequence + [token_id]
        child = Node(new_sequence, node.log_prob_to_node + log_prob, parent=node)
        node.children.append(child)
    return node.children[0] if node.children else node

def simulate(node: Node, max_length: int) -> Tuple[List[int], float]:
    """Simulate a completion from the node."""
    sequence, completion_log_prob = generate_completion(node.sequence, max_length)
    total_quality = node.log_prob_to_node + completion_log_prob
    return sequence, total_quality

def backpropagate(node: Node, quality: float):
    """Update node statistics up the tree."""
    current = node
    while current:
        current.visits += 1
        current.total_quality += quality
        current = current.parent

def mcts(initial_prompt: str, iterations: int, k: int, max_length: int) -> Tuple[str, float]:
    """Run MCTS to generate a sequence."""
    root_sequence = tokenize(initial_prompt)
    root = Node(root_sequence, log_prob_to_node=0.0)
    best_sequence = None
    best_quality = float('-inf')
    
    for _ in range(iterations):
        leaf = select(root)
        if len(leaf.sequence) < max_length:
            child = expand(leaf, k)
            sequence, quality = simulate(child, max_length)
            backpropagate(child, quality)
        else:
            sequence, quality = simulate(leaf, max_length)
            backpropagate(leaf, quality)
        if quality > best_quality:
            best_quality = quality
            best_sequence = sequence
    
    return detokenize(best_sequence) if best_sequence else initial_prompt, best_quality

### Raw Model Generation

def generate_raw_model_sequence(initial_prompt: str, max_length: int) -> Tuple[str, float]:
    """Generate a sequence using greedy decoding."""
    initial_sequence = tokenize(initial_prompt)
    num_to_generate = max_length - len(initial_sequence)
    if num_to_generate <= 0:
        return initial_prompt, 0.0
    sequence, quality = generate_completion(initial_sequence, max_length)
    return detokenize(sequence), quality

### Evaluation Functions

def extract_answer(text: str) -> Optional[str]:
    """Extract the boxed final answer from the generated text."""
    match = re.search(r'\\boxed\{(.*?)\}', text)
    return match.group(1).strip() if match else None

def compare_methods(problem: str, correct_answer: str, iterations: int, k: int, max_length: int) -> Dict[str, bool]:
    """Compare raw model and MCTS on a single problem."""
    prompt = f"Solve the following problem and box your final answer: {problem}"
    
    # Raw model
    raw_text, _ = generate_raw_model_sequence(prompt, max_length)
    raw_answer = extract_answer(raw_text)
    raw_correct = (raw_answer == correct_answer) if raw_answer else False
    
    # MCTS
    mcts_text, _ = mcts(prompt, iterations, k, max_length)
    mcts_answer = extract_answer(mcts_text)
    mcts_correct = (mcts_answer == correct_answer) if mcts_answer else False
    
    return {"raw_correct": raw_correct, "mcts_correct": mcts_correct}

def print_bar(label: str, percentage: float, max_width: int = 10):
    """Print a textual bar chart for accuracy (1 asterisk = 10%)."""
    bar_length = int(percentage / 100 * max_width)
    print(f"{label}: [{'*' * bar_length}] {percentage:.2f}%")

def run_evaluation(problems: List[Dict[str, str]], iterations: int, k: int, max_length: int, sample_size: int = 10):
    """Evaluate both methods on a sample of problems and plot the results."""
    if len(problems) > sample_size:
        problems = random.sample(problems, sample_size)
    print(f"Evaluating {len(problems)} problems from the MATH dataset...")
    
    results = [compare_methods(p["problem"], p["answer"], iterations, k, max_length) for p in problems]
    
    # Calculate accuracies
    total = len(problems)
    raw_correct_count = sum(r["raw_correct"] for r in results)
    mcts_correct_count = sum(r["mcts_correct"] for r in results)
    raw_accuracy = raw_correct_count / total * 100
    mcts_accuracy = mcts_correct_count / total * 100
    
    # Calculate comparison stats
    mcts_better = sum(1 for r in results if r["mcts_correct"] and not r["raw_correct"])
    raw_better = sum(1 for r in results if r["raw_correct"] and not r["mcts_correct"])
    tie = sum(1 for r in results if r["raw_correct"] == r["mcts_correct"])
    
    # Print summary
    print("\n### Evaluation Summary ###")
    print(f"**Total Problems**: {total}")
    print(f"**Raw Model Correct**: {raw_correct_count} ({raw_accuracy:.2f}%)")
    print(f"**MCTS Correct**: {mcts_correct_count} ({mcts_accuracy:.2f}%)")
    print(f"**MCTS Better**: {mcts_better}")
    print(f"**Raw Model Better**: {raw_better}")
    print(f"**Tie**: {tie}")
    
    # Plot accuracies as a textual chart
    print("\n**Accuracy Comparison Chart**")
    print_bar("Raw Model", raw_accuracy)
    print_bar("MCTS", mcts_accuracy)

### Main Execution

if __name__ == "__main__":
    # Simulated MATH dataset (replace with actual dataset loading)
    simulated_problems = [
        {"problem": "What is 2 + 2?", "answer": "4"},
        {"problem": "What is 3 * 5?", "answer": "15"},
        {"problem": "Solve for x: 2x = 10", "answer": "5"},
        {"problem": "What is the area of a square with side 4?", "answer": "16"},
        {"problem": "What is 10 - 7?", "answer": "3"},
        {"problem": "What is the next number in the sequence: 1, 1, 2, 3, 5, ...", "answer": "8"},
        {"problem": "What is 2^3?", "answer": "8"},
        {"problem": "Solve for y: y + 4 = 9", "answer": "5"},
        {"problem": "What is the circumference of a circle with radius 3? Use π=3.14", "answer": "18.84"},
        {"problem": "What is 15 / 3?", "answer": "5"},
    ]
    
    # Configuration parameters
    ITERATIONS = 10  # Number of MCTS iterations
    K = 5           # Top-k tokens to explore
    MAX_LENGTH = 100  # Maximum sequence length
    SAMPLE_SIZE = 5   # Number of problems to evaluate
    
    # Run the evaluation
    run_evaluation(simulated_problems, ITERATIONS, K, MAX_LENGTH, SAMPLE_SIZE)