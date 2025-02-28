import math
import ollama
from typing import List, Tuple, Optional
from transformers import GPT2Tokenizer

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define the Node class for the MCTS tree
class Node:
    def __init__(self, sequence: List[int], parent: Optional['Node'] = None):
        """Initialize a node with a sequence of token IDs."""
        self.sequence = sequence  # List of token IDs
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.total_quality = 0.0  # Sum of quality scores from simulations

# Tokenization functions using the GPT-2 tokenizer
def tokenize(text: str) -> list[int]:
    """Convert text to a list of token IDs."""
    tokens = tokenizer.encode(text)
    print(f"Tokenized '{text}' to {tokens}")
    return tokens

def detokenize(token_ids: list[int]) -> str:
    """Convert a list of token IDs back to text."""
    text = tokenizer.decode(token_ids)
    print(f"Detokenized {token_ids} to '{text}'")
    return text

# Helper functions to interact with Ollama
def get_top_k_tokens(sequence: List[int], k: int) -> List[Tuple[int, float]]:
    """
    Get the top-k next tokens and their log probabilities from the current sequence.
    Returns: List of (token_id, log_prob) tuples.
    """
    prompt = detokenize(sequence)
    print(f"Requesting top-{k} tokens for prompt: '{prompt}'")
    try:
        response = ollama.generate(
            model='deepseek-r1:14b',
            prompt=prompt,
            options={
                'num_predict': 1,  # Generate one token
                'top_k': k,
                'logprobs': True   # Request log probabilities
            }
        )
        print(f"Ollama response for top-k: {response}")
        # Assuming 'logprobs' is a list of {'token': token_id, 'logprob': log_prob}
        logprobs = response.get('logprobs', [])
        if not logprobs:
            print("Warning: No logprobs returned from Ollama")
            return []
        sorted_logprobs = sorted(logprobs, key=lambda x: x['logprob'], reverse=True)[:k]
        result = [(item['token'], item['logprob']) for item in sorted_logprobs]
        print(f"Top-{k} tokens: {result}")
        return result
    except Exception as e:
        print(f"Error in get_top_k_tokens: {e}")
        return []

def generate_completion(sequence: List[int], max_length: int) -> Tuple[List[int], float]:
    """
    Generate a completion from the sequence up to max_length tokens.
    Returns: (completed_sequence, total_log_prob).
    """
    prompt = detokenize(sequence)
    num_to_generate = max_length - len(sequence)
    print(f"Generating completion for '{prompt}', num_to_generate: {num_to_generate}")
    if num_to_generate <= 0:
        print(f"Sequence already at or above max_length: '{prompt}'")
        return sequence, 0.0  # No generation needed
    try:
        response = ollama.generate(
            model='deepseek-r1:14b',
            prompt=prompt,
            options={
                'num_predict': num_to_generate,
                'temperature': 0.0  # Greedy decoding for deterministic simulation
            }
        )
        print(f"Ollama response for completion: {response}")
        # Assuming 'response' contains the generated text
        generated_text = response.get('response', '')
        print(f"Generated text: '{generated_text}'")
        # Tokenize the generated text and append to the sequence
        generated_tokens = tokenize(generated_text)
        completed_sequence = sequence + generated_tokens
        # Assuming logprobs are provided for each token
        logprobs = response.get('logprobs', [])
        total_log_prob = sum(logprobs) if logprobs else 0.0
        print(f"Log probabilities: {logprobs}, Total log prob: {total_log_prob}")
        return completed_sequence, total_log_prob
    except Exception as e:
        print(f"Error in generate_completion: {e}")
        return sequence, 0.0

# MCTS components
def ucb(node: Node, C: float = 1.0) -> float:
    """Compute the UCB1 score for a node."""
    if node.visits == 0:
        return float('inf')  # Prioritize unvisited nodes
    parent_visits = node.parent.visits if node.parent else 1
    score = (node.total_quality / node.visits) + C * math.sqrt(math.log(parent_visits) / node.visits)
    print(f"UCB for '{detokenize(node.sequence)}': visits={node.visits}, total_quality={node.total_quality}, score={score}")
    return score

def select(root: Node) -> Node:
    """Select a leaf node using UCB."""
    current = root
    print(f"Starting selection from root: '{detokenize(root.sequence)}'")
    while current.children:
        current = max(current.children, key=lambda n: ucb(n))
        print(f"Selected child: '{detokenize(current.sequence)}'")
    print(f"Reached leaf: '{detokenize(current.sequence)}'")
    return current

def expand(node: Node, k: int) -> Node:
    """Expand a node by adding top-k next tokens as children and return the best child."""
    top_k = get_top_k_tokens(node.sequence, k)
    print(f"Expanding node '{detokenize(node.sequence)}' with {len(top_k)} children: {top_k}")
    for token_id, log_prob in top_k:
        new_sequence = node.sequence + [token_id]
        child = Node(new_sequence, parent=node)
        node.children.append(child)
        print(f"Added child: '{detokenize(new_sequence)}', log_prob={log_prob}")
    # Select the child with the highest log_prob to simulate (first child, as top_k is sorted)
    return node.children[0] if node.children else node

def simulate(node: Node, max_length: int) -> Tuple[List[int], float]:
    """Simulate a completion from the node and return sequence and quality."""
    print(f"Simulating from node: '{detokenize(node.sequence)}'")
    sequence, quality = generate_completion(node.sequence, max_length)
    print(f"Simulation result: '{detokenize(sequence)}', Quality: {quality}")
    return sequence, quality

def backpropagate(node: Node, quality: float):
    """Update node statistics up the tree with the simulation quality."""
    current = node
    while current:
        current.visits += 1
        current.total_quality += quality
        print(f"Backpropagating to '{detokenize(current.sequence)}': visits={current.visits}, total_quality={current.total_quality}")
        current = current.parent

def mcts(initial_prompt: str, iterations: int, k: int, max_length: int) -> str:
    """
    Run MCTS to generate a high-quality sequence.
    
    Args:
        initial_prompt: Starting text (e.g., a chain-of-thought prompt).
        iterations: Number of MCTS iterations.
        k: Number of children to expand per node.
        max_length: Maximum sequence length in tokens.
    
    Returns:
        The best generated sequence as text.
    """
    print(f"\nStarting MCTS with prompt: '{initial_prompt}', iterations={iterations}, k={k}, max_length={max_length}")
    root_sequence = tokenize(initial_prompt)
    root = Node(root_sequence)
    best_sequence = None
    best_quality = float('-inf')
    
    for i in range(iterations):
        print(f"\n--- Iteration {i + 1}/{iterations} ---")
        leaf = select(root)
        print(f"Selected leaf: '{detokenize(leaf.sequence)}', length={len(leaf.sequence)}")
        if len(leaf.sequence) < max_length:
            child = expand(leaf, k)
            sequence, quality = simulate(child, max_length)
            backpropagate(child, quality)
        else:
            print("Leaf at max_length; simulating without expansion")
            sequence, quality = simulate(leaf, max_length)
            backpropagate(leaf, quality)
        # Update best sequence if this one has higher quality
        if quality > best_quality:
            best_quality = quality
            best_sequence = sequence
            print(f"New best sequence found: '{detokenize(best_sequence)}', Quality: {best_quality}")
    
    final_sequence = detokenize(best_sequence) if best_sequence else initial_prompt
    print(f"\nMCTS completed. Final best sequence: '{final_sequence}', Best quality: {best_quality}")
    return final_sequence

# Example usage
if __name__ == "__main__":
    # Example chain-of-thought prompt
    prompt = "Prove the fermat hypotesys. no shortcuts."
    
    # Parameters
    ITERATIONS = 50    # Number of iterations for MCTS exploration
    K = 10             # Number of children per expansion
    MAX_LENGTH = 200   # Maximum sequence length in tokens
    
    # Run MCTS
    result = mcts(prompt, ITERATIONS, K, MAX_LENGTH)
    print("\nFinal Output:")
    print(result)