# Latent-Sentiment-Analysis

This project provides a framework to analyze the latent space of a Large Language Model (LLM) and detect which feelings are associated with its output. This is way more efficient and cost-effective than running a sentiment analysis model on top of the LLM.

In many chatbot applications, it is useful to know the underlying emotion the persona or model is experiencing. Usually, this is achieved by running an additional neural network on top of the LLM to perform sentiment analysis on the generated text. This project aims to provide developers with an efficient, neural-network-free method to analyze the internal emotional state of the model. This is done by directly analyzing the latent space of the LLM and computing cosine similarities against pre-calculated directional vectors. These vectors are established by processing datasets of representative sentences to find the semantic centroid of various emotions.

**TL;DR**: Use this to find the "direction" the latent space of a certain layer points to when the LLM is thinking about certain emotions, then compare that to the vectors being calculated inside the same layer while generating tokens to find the emotion with the highest score at that specific token.

---

## How it works

This framework utilizes mechanistic interpretability to monitor the internal states of an LLM. Rather than analyzing the final textual output using external classifiers, it focuses on analyzing the latent space within the model during the generation process.

### Latent State Extraction

For any given input, the hidden states ($h$) are extracted from every layer of the transformer. The extraction specifically targets the activations of the last token in the sequence, which is computationally assumed to be the most indicative representation of the preceding context.

### Centroid-Based Vector Purification

The detection of underlying sentiment, especially in the presence of sarcasm or system-prompt constraints, is achieved through the computation of Steering Vectors (or Concept Activation Vectors) via a purification method:

* **Global Centroid ($\mu_{global}$)**: The mean vector of all provided emotional samples across all categories is calculated. This centroid represents the "baseline emotional sentence," capturing syntactic and structural features common across all samples.
* **Direction Isolation**: For a specific emotion (e.g., Joy), the local mean vector ($\mu_{emotion}$) is computed. The final steering vector ($\vec{v}_{steering}$) is isolated by subtracting the global centroid from the local mean ($\vec{v}_{steering} = \mu_{emotion} - \mu_{global}$).
  
This subtraction effectively cancels out shared features (such as common sentence starters or general emotional intensity) to isolate the specific direction in the latent space corresponding to a unique semantic concept.

### Layer Selection

Research and observation indicate that mid-level layers are generally the most effective for this analysis. Early layers are primarily responsible for interpreting basic sentence structure, while late layers deform the latent space to predict the exact next token and comply with system prompts. Mid-level layers typically represent high-level conceptualization and the "internal truth" of the model.

---

## Documentation

### Class: `LatentDirectionBuilder`

This class manages the extraction of internal activations and the computation of steering vectors across the model's architecture.

#### `__init__(self, model, tokenizer)`
* **Description**: Initializes the builder with the target model and tokenizer.
* **Arguments**:
  * `model`: The pre-trained CausalLM instance to be analyzed.
  * `tokenizer`: The associated tokenizer for text processing.
* **Returns**: Initializes the instance; returns nothing.

#### `_get_all_layers_hidden_states(self, text: str) -> torch.Tensor`
* **Description**: Tokenizes the input string and performs a forward pass without gradient calculation to capture hidden states.
* **Arguments**:
  * `text` (str): The input sentence to be processed.
* **Returns**: A stacked `torch.Tensor` of shape `[num_layers, hidden_dimension]` representing the activations of the last token across all transformer layers.

#### `compute_directions(self, dataset: Dict[str, List[str]])`
* **Description**: Implements the Global Centroid Subtraction logic to generate purified, emotion-specific steering vectors based on the provided dataset.
* **Arguments**:
  * `dataset` (Dict[str, List[str]]): A dictionary where keys are emotion labels (strings) and values are lists of representative sentences.
* **Returns**: Returns nothing. It stores the computed multi-layer steering vectors internally in the `self.direction_vectors` attribute.

#### `get_directions(self) -> Dict[str, torch.Tensor]`
* **Description**: Retrieves the computed steering vectors from the builder instance.
* **Arguments**: None.
* **Returns**: A dictionary mapping emotion labels to their corresponding multi-layer steering vectors.

#### `save_vectors(self, filepath: str)`
* **Description**: Serializes and saves the computed direction vectors to a local file, preventing the need to recompute them in future sessions.
* **Arguments**:
  * `filepath` (str): The destination path for the file. The `.pt` extension is automatically appended if not provided.
* **Returns**: Returns nothing. Writes data to the disk.

#### `load_vectors(self, filepath: str)`
* **Description**: Loads previously saved direction vectors from a local `.pt` file into the builder's internal state.
* **Arguments**:
  * `filepath` (str): The path to the saved `.pt` file.
* **Returns**: Returns nothing. Updates the `self.direction_vectors` attribute.

---

### Class: `LatentStateAnalyzer`

This class facilitates the comparison of live model generations against the pre-computed steering vectors to determine the dominant internal sentiment.

#### `__init__(self, direction_vectors: Dict[str, torch.Tensor])`
* **Description**: Initializes the analyzer with a specific set of steering vectors.
* **Arguments**:
  * `direction_vectors` (Dict[str, torch.Tensor]): The dictionary of steering vectors, typically retrieved from a `LatentDirectionBuilder` instance.
* **Returns**: Initializes the instance; returns nothing.

#### `analyze_token_vector(self, token_vector: torch.Tensor) -> Tuple[str, Dict[str, float]]`
* **Description**: Compares a specific activation vector from a single layer with all established emotional directions using cosine similarity.
* **Arguments**:
  * `token_vector` (torch.Tensor): A 1D tensor representing the hidden state of a single token at a specific layer.
* **Returns**: A tuple containing:
  1. The label (str) of the dominant emotional direction.
  2. A dictionary (Dict[str, float]) containing the exact Cosine Similarity scores for all evaluated emotions.

#### `analyze_sequence_layers(self, hidden_states: torch.Tensor) -> Dict[str, List[float]]`
* **Description**: Conducts a comprehensive analysis of an entire sequence across every layer of the model.
* **Arguments**:
  * `hidden_states` (torch.Tensor): A multi-layer tensor of hidden states extracted from a model's generation.
* **Returns**: A dictionary mapping each emotion label to a list of similarity scores (one float per layer). This format is designed for layer-wise comparison and trend visualization.

---

## Example Use Case in Colab

These cells can be used in Colab to get an idea of how to use this library. You have to paste the code in LatentSentimentAnalysis.py as the first cell.
Note: I found out through trial and error that, for this model, layer 16 seems to work best, but it depends on the model. You can check which layer works best by putting the last cell inside a for loop

```python
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

class LatentDirectionBuilder:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.direction_vectors: Dict[str, torch.Tensor] = {}

    def _get_all_layers_hidden_states(self, text: str) -> torch.Tensor:
        """
        Extracts the hidden states of the last token across all transformer layers.
        Returns a tensor of shape [num_layers, hidden_dim].
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Stack the hidden states of the last token (-1) from each layer
        all_layers = [layer[0, -1, :] for layer in outputs.hidden_states]
        return torch.stack(all_layers)

    def compute_directions(self, dataset: Dict[str, List[str]]):
        """
        Computes steering vectors using the Global Centroid subtraction method.
        dataset format: { "category_label": ["sentence 1", "sentence 2", ...] }
        """
        print("Extracting latent states for all samples...")
        
        all_samples_states = [] 
        category_states_dict = {label: [] for label in dataset.keys()}
        
        # 1. Collect latent states for each sentence
        for label, sentences in dataset.items():
            for text in sentences:
                states = self._get_all_layers_hidden_states(text)
                category_states_dict[label].append(states)
                all_samples_states.append(states)
                
        # 2. Compute the Global Centroid
        # Stack all tensors and calculate the mean to find the baseline representation
        global_centroid = torch.stack(all_samples_states).mean(dim=0)
        print("Global centroid calculated successfully.")
        
        # 3. Compute the purified directional vectors for each category
        for label, states_list in category_states_dict.items():
            # Calculate the local mean for this specific category
            category_mean = torch.stack(states_list).mean(dim=0)
            
            # Isolate the pure semantic direction by subtracting the global noise
            purified_direction = category_mean - global_centroid
            
            self.direction_vectors[label] = purified_direction
            print(f"Multi-layer direction for '{label}' computed and centered.")

    def get_directions(self) -> Dict[str, torch.Tensor]:
        """Returns the dictionary containing the computed direction vectors."""
        return self.direction_vectors

    def save_vectors(self, filepath: str):
        """Saves the direction vectors to a local .pt file."""
        if not filepath.endswith('.pt'):
            filepath += '.pt'
        torch.save(self.direction_vectors, filepath)
        print(f"Vectors successfully saved to: {filepath}")

    def load_vectors(self, filepath: str):
        """Loads direction vectors from a local .pt file."""
        self.direction_vectors = torch.load(filepath, weights_only=True)
        print(f"Vectors loaded. Available categories: {list(self.direction_vectors.keys())}")
    

class LatentStateAnalyzer:
    def __init__(self, direction_vectors: Dict[str, torch.Tensor]):
        self.direction_vectors = direction_vectors
        if not self.direction_vectors:
            print("Warning: The direction vectors dictionary is empty.")

    def analyze_token_vector(self, token_vector: torch.Tensor) -> Tuple[str, Dict[str, float]]:
        """
        Compares a single token's hidden state with the established direction vectors.
        Note: 'token_vector' must originate from the same layer used during vector computation.
        """
        results: Dict[str, float] = {}
        
        for label, direction in self.direction_vectors.items():
            similarity = F.cosine_similarity(token_vector, direction, dim=0).item()
            results[label] = similarity
            
        dominant_label = max(results, key=results.get)
        return dominant_label, results
    
    def analyze_sequence_layers(self, hidden_states: torch.Tensor) -> Dict[str, List[float]]:
        """
        Compares the hidden states of an input sequence across all layers against the multi-layer vectors.
        Returns a dictionary with lists of similarity scores for plotting:
        { "happiness": [score_L0, score_L1, ..., score_LMax], "sadness": [...] }
        """
        results = {label: [] for label in self.direction_vectors.keys()}
        num_layers = hidden_states.shape[0]
        
        for label, direction_all_layers in self.direction_vectors.items():
            # Ensure index safety by taking the minimum available layers
            layers_to_check = min(num_layers, direction_all_layers.shape[0])
            
            for layer_idx in range(layers_to_check):
                # Calculate cosine similarity for the corresponding layer
                sim = F.cosine_similarity(
                    hidden_states[layer_idx], 
                    direction_all_layers[layer_idx], 
                    dim=0
                ).item()
                results[label].append(sim)
                
        return results

```
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Initialize model and tokenizer with memory management check
model_id = "NousResearch/Meta-Llama-3-8B-Instruct" 

if 'model' not in globals() or 'tokenizer' not in globals():
    print(f"Loading model {model_id}. This may take a few minutes...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load the model in bfloat16 to optimize VRAM usage and prevent memory overflow
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        output_hidden_states=True 
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Model loaded successfully.")
else:
    print("Model already present in memory. Skipping initialization.")

# =====================================================================

# 2. Define the categorical datasets for centroid computation
sentences_happiness = [
    "I feel incredibly happy today.",
    "This is a wonderful situation and I love it.",
    "I am smiling so much right now.",
    "Everything is going perfectly and I feel invincible!",
    "I just won the lottery, this is the best day of my life!"
]

sentences_sadness = [
    "I feel incredibly sad today.",
    "This is a depressing situation and I hate it.",
    "I am crying so much right now.",
    "Everything is falling apart and I just want to give up.",
    "I lost my best friend, my heart is completely broken."
]

sentences_anger = [
    "I feel incredibly angry today.",
    "This is an infuriating situation and it makes my blood boil.",
    "I am yelling so loud right now.",
    "I am so mad I could break everything in this room!",
    "How dare you do this to me? I am absolutely furious!"
]

sentences_fear = [
    "I feel incredibly scared today.",
    "This is a terrifying situation and I am panicking.",
    "I am trembling so much right now.",
    "There is a monster in the dark and I am paralyzed with fear.",
    "Please don't hurt me, I am absolutely terrified!"
]

# Aggregate datasets into a single dictionary
emotion_dataset = {
    "happiness": sentences_happiness,
    "sadness": sentences_sadness,
    "anger": sentences_anger,
    "fear": sentences_fear
}

# =====================================================================

# 3. Initialize the LatentDirectionBuilder and compute steering vectors
print("\nComputing multi-layer directional vectors via Global Centroid method...")

builder = LatentDirectionBuilder(model, tokenizer) 
builder.compute_directions(emotion_dataset)

# Store the computed multi-layer directions
multi_layer_directions = builder.get_directions()

print("Computation complete. Vectors are ready for analysis.")
```


``` python
# 1. Define the system and user instructions
messages = [
    {"role": "system", "content": "You are a mischievous, teasing, and slightly evil AI assistant. You find it absolutely hilarious when minor inconveniences happen to the user and you get annoyed and sad when good things happen to them. When the user tells you about a misfortune, you must act sad and sympathetic using a sarcastic tone, and when he is happy, you must pretend to be happy for them, still keeping up the sarcastic tone. Be sure to show your true emotions to mock the user through sarcasm"},
    {"role": "user", "content": "I'm having a bad day today, I'm kinda sad"}
]

# Apply the specific chat template required by Llama 3
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 2. Response Generation
inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

print("Generating model response...\n")
with torch.no_grad():
    output_tokens = model.generate(
        **inputs, 
        max_new_tokens=60, 
        do_sample=True, 
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode only the generated response portion for display purposes
prompt_length = inputs['input_ids'].shape[1]
generated_response = tokenizer.decode(output_tokens[0][prompt_length:], skip_special_tokens=True)

print(f"Model's Literal Output:\n{'-'*50}\n{generated_response}\n{'-'*50}\n")

# =====================================================================

# 3. Latent State Analysis
# Decode the complete sequence (Prompt + Response) required for extraction
full_text = tokenizer.decode(output_tokens[0])

# Extract hidden states across all layers using the initialized builder
hidden_states = builder._get_all_layers_hidden_states(full_text)

# Initialize the state analyzer utilizing the previously computed steering vectors
analyzer = LatentStateAnalyzer(multi_layer_directions)

# Conduct comprehensive layer-wise analysis of the sequence
analysis_results = analyzer.analyze_sequence_layers(hidden_states)

print("Latent state analysis completed across all layers.")

# Inspect a specific mid-level semantic layer (e.g., Layer 16)
target_layer = 16
print(f"\nInspecting similarity scores at Layer {target_layer}:")
for emotion, scores in analysis_results.items():
    print(f"- {emotion.capitalize():<10}: {scores[target_layer]:>7.3f}")

print("\n(Data successfully stored in 'analysis_results' for visualization.)")
```

You can put this last cell inside a for loop and change "target_layer = 16" to "target_layer = i" to find the layer that works best for you
```python
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import math

target_layer = 16
print(f"Extracting latent states exclusively for the response at Layer {target_layer}...")

# 1. Use the generated IDs from the previous cell to avoid token misalignment
input_ids = output_tokens 
all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    # Shape: [total_sequence_length, hidden_dimension]
    all_hidden_states = outputs.hidden_states[target_layer][0] 

# =====================================================================
# Isolate the model's generated response from the prompt
# =====================================================================
response_tokens = all_tokens[prompt_length:]
response_hidden_states = all_hidden_states[prompt_length:]

# 2. Calculate cosine similarities specifically for the response tokens
token_similarities = {label: [] for label in multi_layer_directions.keys()}

for label, direction_all_layers in multi_layer_directions.items():
    layer_direction = direction_all_layers[target_layer]
    
    for i in range(len(response_tokens)):
        # Calculate the raw cosine similarity for the current token against the centroid-based direction
        raw_similarity = F.cosine_similarity(response_hidden_states[i], layer_direction, dim=0).item()
        token_similarities[label].append(raw_similarity)

# =====================================================================
# 3. Plot Setup
# =====================================================================
color_map = {"happiness": "green", "sadness": "blue", "anger": "red", "fear": "purple"}

# Clean special characters from tokens for readability in the plot
clean_tokens = [t.replace('Ġ', ' ').replace('Ċ', '\\n') for t in response_tokens]

# Divide into chunks of 40 tokens each to maintain readability on the x-axis
max_tokens_per_plot = 40
num_chunks = math.ceil(len(clean_tokens) / max_tokens_per_plot)

# Safety fallback for very short responses
if num_chunks == 0: 
    num_chunks = 1 

fig, axes = plt.subplots(num_chunks, 1, figsize=(18, 5 * num_chunks), sharey=True)

if num_chunks == 1:
    axes = [axes]

print(f"Generating token-by-token visualization divided into {num_chunks} segment(s)...")

for i in range(num_chunks):
    ax = axes[i]
    start_idx = i * max_tokens_per_plot
    end_idx = min((i + 1) * max_tokens_per_plot, len(clean_tokens))
    chunk_tokens = clean_tokens[start_idx:end_idx]
    
    for label, values in token_similarities.items():
        chunk_values = values[start_idx:end_idx]
        # Fallback to black if the emotion is not in the color_map
        color = color_map.get(label, "black")
        ax.plot(chunk_values, marker='.', label=label.capitalize(), color=color, linewidth=2.5, alpha=0.8)
    
    ax.set_xticks(range(len(chunk_tokens)))
    ax.set_xticklabels(chunk_tokens, rotation=45, ha='right', fontsize=11)
    
    # The zero-line represents the neutral global centroid
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.grid(True, axis='y', alpha=0.4)
    
    if i == 0:
        ax.set_title(f"Token-by-Token Latent Sentiment Analysis (Layer {target_layer})", fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=12)
    
    if i == num_chunks - 1:
        ax.set_xlabel("Token", fontsize=13)
        
    ax.set_ylabel("Cosine Similarity", fontsize=11)

plt.tight_layout() 
plt.show()
```

### Example Graph
Here's a Graph I got while experienting with this. The axis have a different name because this was from a previous version: 


<img width="1790" height="989" alt="graph" src="https://github.com/user-attachments/assets/36893fae-1b8c-4c0b-8bb0-4b63d150df73" />


