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
        """Extracts the hidden states of the last token for all layers. Returns [num_layers, hidden_dim]."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Stack the last token's hidden state from each layer
        all_layers = [layer[0, -1, :] for layer in outputs.hidden_states]
        return torch.stack(all_layers)

    def compute_directions(self, dataset: Dict[str, List[str]]):
        """
        Computes steering vectors using the 'Center of Mass' (Mean Difference) method.
        dataset: { "emotion_name": ["sentence_1", "sentence_2", ...] }
        """
        print("🔄 Extracting latent states for all samples...")
        
        all_samples_states = [] 
        emotion_samples_dict = {label: [] for label in dataset.keys()}
        
        for label, sentences in dataset.items():
            for text in sentences:
                states = self._get_all_layers_hidden_states(text)
                emotion_samples_dict[label].append(states)
                all_samples_states.append(states)
                
        # Calculate Global Centroid (the average latent representation of all samples)
        global_centroid = torch.stack(all_samples_states).mean(dim=0)
        print("🌍 Global centroid calculated.")
        
        # Calculate purified steering vectors for each category
        for label, states_list in emotion_samples_dict.items():
            # Local mean for the specific category
            category_mean = torch.stack(states_list).mean(dim=0)
            
            # Isolate the direction by subtracting the global mean (noise reduction)
            purified_direction = category_mean - global_centroid
            
            self.direction_vectors[label] = purified_direction
            print(f"✅ Multi-layer direction for '{label}' computed and centered.")

    def get_directions(self) -> Dict[str, torch.Tensor]:
        return self.direction_vectors

    def save_vectors(self, filepath: str):
        if not filepath.endswith('.pt'):
            filepath += '.pt'
        torch.save(self.direction_vectors, filepath)
        print(f"💾 Vectors saved to: {filepath}")

    def load_vectors(self, filepath: str):
        self.direction_vectors = torch.load(filepath, weights_only=True)
        print(f"📂 Vectors loaded. Categories: {list(self.direction_vectors.keys())}")


class LatentStateAnalyzer:
    def __init__(self, direction_vectors: Dict[str, torch.Tensor]):
        self.direction_vectors = direction_vectors
        if not self.direction_vectors:
            print("⚠️ Warning: Direction vectors dictionary is empty!")

    def analyze_token_vector(self, token_vector: torch.Tensor) -> Tuple[str, Dict[str, float]]:
        """
        Compares a single token vector with pre-computed directions.
        Note: token_vector must originate from the same layer as the directions.
        """
        results: Dict[str, float] = {}
        
        for label, direction in self.direction_vectors.items():
            similarity = F.cosine_similarity(token_vector, direction, dim=0).item()
            results[label] = similarity
            
        top_direction = max(results, key=results.get)
        return top_direction, results
    
    def analyze_sequence_layers(self, hidden_states: torch.Tensor) -> Dict[str, List[float]]:
        """
        Compares input hidden states across all layers with pre-computed directions.
        Returns a dictionary with similarity scores for each category per layer.
        """
        results = {label: [] for label in self.direction_vectors.keys()}
        num_layers = hidden_states.shape[0]
        
        for label, direction_all_layers in self.direction_vectors.items():
            # Ensure we don't exceed the available layer count
            layers_to_check = min(num_layers, direction_all_layers.shape[0])
            
            for layer_idx in range(layers_to_check):
                # Cosine similarity between layer N of input and layer N of the direction
                sim = F.cosine_similarity(
                    hidden_states[layer_idx], 
                    direction_all_layers[layer_idx], 
                    dim=0
                ).item()
                results[label].append(sim)
                
        return results