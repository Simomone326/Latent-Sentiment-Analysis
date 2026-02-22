# Latent-Sentiment-Analysis
this project provides a framework to analyze the latent space of an LLM and detect which feelings are associated to the output.

In many chatbots it may be useful to know which emotion the character that the model is roleplaying is feeling. 
Usually, this is done by running an additional neural network on top of the LLM that does sentiment analysis on the output. 
This project aims to provide developers with an efficient, neural network free method to analyze the feelings of the chatbot by directly using the latent space of the LLM and doing a cosine similarity with pre-calculated vectors, which are calculated by providing a list of sentences that rapresent a certain emotion (more than one emotion is required as the model needs to find the centroid of all the different emotions)


Ecco la versione del **Technical Deep Dive** e della **Documentation** riscritta con un tono impersonale e accademico, ideale per un repository GitHub professionale.

---

## How it works

This framework utilizes Mechanicistic interpretability to monitor the internal states of a Large Language Model (LLM). Rather than analyzing the final textual output using another neural network, it focuses on analyzing the latent space on the model during the generation of the output.

### Latent State Extraction

For any given input, the hidden states () are extracted from every layer of the transformer. The extraction specifically targets the activations of the last token in the sequence, which is assumed to be the most indicative of the previous context

### Centroid-Based Vector Purification

The detection of underlying sentiment, especially in the presence of sarcasm or system-prompt constraints, is achieved through the computation of **Steering Vectors** (or Concept Activation Vectors) via a purification method:

* **Global Centroid ()**: The mean vector of all provided emotional samples is calculated. This centroid represents the "baseline emotional sentence," capturing syntactic and structural features common across all samples.
* **Direction Isolation**: For a specific emotion (e.g., Joy), the local mean vector () is computed. The steering vector is then isolated by subtracting the global centroid.
  This subtraction effectively cancels out shared features (such as common sentence starters or general emotional intensity) to isolate the specific direction in the latent space corresponding to a unique semantic concept.

### What layer is the best?

The best layers are usually the middle ones, which research suggests are the ones responsible for high-level conceptualization, while the early ones are responsible for interpreting the sentence structure and late ones for predicting the next token 

---

## Documentation

### Class: `LatentDirectionBuilder`

This class manages the extraction of activations and the computation of steering vectors across the model's architecture.

#### `__init__(self, model, tokenizer)`

* **model**: The pre-trained CausalLM to be analyzed.
* **tokenizer**: The associated tokenizer for text processing.

#### `_get_all_layers_hidden_states(self, text: str) -> torch.Tensor`

Tokenizes the input and performs a forward pass to capture hidden states.
**Returns** A stacked tensor of shape `[num_layers, hidden_dimension]` representing the activations of the last token across all layers.

#### `compute_directions(self, dataset: Dict[str, List[str]])`

Implements the Global Centroid Subtraction logic to generate purified vectors.
* **Arguments**: A dictionary where keys are emotion labels and values are lists of representative sentences.
* **Result**: Stores the purified multi-layer steering vectors in `self.direction_vectors`.

---

### Class: `LatentStateAnalyzer`

This class facilitates the comparison of live model generations against the pre-computed steering vectors.

#### `__init__(self, direction_vectors: Dict[str, torch.Tensor])`

* **direction_vectors**: The dictionary of steering vectors produced by a `LatentDirectionBuilder` instance.

#### `analyze_token_vector(self, token_vector: torch.Tensor) -> Tuple[str, Dict[str, float]]`

Compares a specific activation vector (from a single layer) with all established directions.
**Returns** the label of the dominant direction and a dictionary containing all Cosine Similarity scores.

#### `analyze_sequence_layers(self, hidden_states: torch.Tensor) -> Dict[str, List[float]]`

 Conducts a comprehensive analysis of a sequence across every layer of the model.
* **Arguments**: A tensor of hidden states extracted from a model response.
* **Returns** A dictionary mapping each emotion label to a list of similarity scores (one per layer), suitable for trend visualization and layer-wise comparison.

---


