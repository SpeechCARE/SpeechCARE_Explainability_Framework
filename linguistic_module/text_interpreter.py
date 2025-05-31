import openai  # You can replace this with any other LLM API client
from typing import Dict, List, Union
from modelscope import AutoTokenizer
from modelscope import AutoModelForCausalLM
import torch
import numpy as np
import json

system_prompt1 = """
    You are a specialized language model trained to detect linguistic cues of cognitive impairment. You will receive:
    1) A set of linguistic features to consider.
    2) A text passage to analyze.
    3) Token-level SHAP values from a pre-trained model.
    
    ---
    ## Linguistic Features to Consider:
    • Lexical Richness: Unusual or varied vocabulary, overuse of vague terms (e.g., “thing,” “stuff”).
    • Syntactic Complexity: Simple vs. complex sentence constructions, grammatical errors.
    • Sentence Length and Structure: Fragmented vs. compound/complex sentences.
    • Repetition: Repeated words, phrases, or clauses.
    • Disfluencies and Fillers: Terms like “um,” “uh,” “like.”
    • Semantic Coherence and Content: Logical flow of ideas, clarity of meaning.
    • Additional Feature: Placeholder for any extra marker (e.g., specialized domain terms).
    ---
    ## Text to Analyze:
    {text}
    ---
    ## Token-level SHAP Values:
    {shap_values}
    ---
    You must analyze the given text and the shap values based on:
    Synthesize the significance of provided tokens/features to explain how they collectively point to healthy cognition or potential cognitive impairment.
    Ensure that the explanations are concise, insightful, and relevant to cognitive impairment assessment.
    Output should be structured as **bullet points**, with each bullet clearly describing one key aspect of the analysis. 
    ---
    ## Analysis:
    
    """

system_prompt2 = """
    Based on the the long analysis of detecting cognitive impairment, provide the following:
    Write the final prediction regarding the detection of cognitive impairment in **one short sentence**.
    Summarize the key findings and their implications in **bullet points**, without using the "Title: description" format.
    Do not provide any additional or extra explanations and points.
    **Avoid saying anything about SHAP values and Giving Suggestions for further analysis**
    **Do not repeat yourself**
    ---
    ## Long Analysis:
    {generated_text}

    ---
    ## Final Prediction and Key findingd:
    
    """
class TextInterpreter:
    """
    A class to interpret SHAP values and analyze text for cognitive impairment cues using an LLM.
    """
    
    def __init__(self,model_path: str ='/workspace/models/llama70B'):
        """
        Initialize the interpreter with the LLM API key and model.
        
        Args:
            model_path (str): Path to the LLM model to use.
        """
        self.model,self.tokenizer = self.initialize_model(model_path)

    def initialize_model(self, model_path ):
        """
        Initializes and returns the language model and tokenizer with optimized settings.
        
        Returns:
            tuple: (model, tokenizer) pair for text generation
        """
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
        return model, tokenizer
    def format_shap_values(self,shap_explanation):
        """
        Convert SHAP Explanation object to list of (token, SHAP value) pairs.
        
        Args:
            shap_explanation: SHAP Explanation object
            
        Returns:
            list: List of tuples in format (token, shap_value)
        """
        # Get tokens and values
        tokens = np.array(shap_explanation.data[0])  # Convert to numpy array if not already
        values = shap_explanation.values
            
        # Create (token, value) pairs
        token_value_pairs = []
        for token, value in zip(tokens, values):
            token_str = str(token)
            # Handle scalar values (single classification) or arrays (multi-class)
            shap_value = float(value) if np.isscalar(value) else [float(v) for v in value]
            token_value_pairs.append((token_str, shap_value))
        
        return token_value_pairs
   
    def analyze_cognitive_impairment(self, transcription: str, shap_values: Union[Dict, List],shap_index:int) -> str:
        """
        Generates the initial linguistic analysis using the provided transcription and SHAP values.
        
        Args:
            transcription (str): Text to analyze
            shap_values (Union[Dict, List]): SHAP values for interpretation
            
        Returns:
            str: The generated analysis text
        """
        shap_values.values = shap_values[0,:,shap_index]
        token_shap_pairs = self.format_shap_values(shap_values) 

        prompt = system_prompt1.format(text=transcription, shap_values=json.dumps(token_shap_pairs, indent=2))
        
        # Call the LLM
        response = self._call_llm(prompt)
        return response
    
    def refine_analysis(
        self, 
        previous_response: str, 
        refinement_metrics: Dict[str, float]
    ) -> str:
        """
        Refine the previous analysis using precomputed linguistic/cognitive metrics.
        
        Args:
            previous_response (str): LLM's previous analysis of the text.
            refinement_metrics (Dict[str, float]): Precomputed metrics for the text.
                Example: {'lexical_richness': 0.5, 'syntactic_complexity': 0.7}
                - Values are scores (e.g., 0.5 = low lexical richness, 0.9 = high).
                
        Returns:
            str: Refined analysis incorporating the metrics.
        """
        # Format the metrics for the prompt
        metrics_description = "\n".join(
            [f"- {metric.replace('_', ' ').title()}: {value} (Range: 0=low, 1=high)" 
             for metric, value in refinement_metrics.items()]
        )
        
        input_message = (
            f"### Previous Analysis ###\n{previous_response}\n\n"
            f"### New Linguistic/Cognitive Metrics ###\n"
            f"The following metrics were computed for the text (0=low, 1=high):\n"
            f"{metrics_description}\n\n"
            f"### Refinement Task ###\n"
            f"Revise the previous analysis by incorporating these metrics:\n"
            f"1. **Lexical Richness**: Adjust interpretation based on vocabulary diversity. "
            f"Low scores suggest repetitive or simple word use; high scores indicate varied vocabulary.\n"
            f"2. **Syntactic Complexity**: Update analysis of sentence structure. "
            f"Low scores imply short/grammatically simple sentences; high scores suggest complex syntax.\n"
            f"3. **Semantic Coherence**: Reassess logical flow. "
            f"Low scores indicate tangential/incoherent speech; high scores reflect clear logic.\n"
            f"4. **Memory Indicators**: Reinterpret mentions of forgetfulness. "
            f"Low scores may imply weaker evidence; high scores strengthen the case.\n\n"
            f"### Output Format ###\n"
            f"- Compare the metrics to typical cognitive impairment patterns.\n"
            f"- Explain how the scores support or contradict the initial analysis.\n"
            f"- Highlight any new insights (e.g., 'Low lexical richness aligns with expected decline in vocabulary').\n"
        )
        
        return self._call_llm(input_message)
    
    def prep_prompt_summarize(self,generated_text):
        content = system_prompt2.format(generated_text=generated_text)
        messages = [{"role": "user", "content": content}]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return prompt
    
    def generate_final_interpretation(self,analysis_text: str) -> str:
        """
        Generates a final interpretation based on the previous analysis of text.
        
        Args:
            analysis_text (str): The initial analysis to summarize
            
        Returns:
            str: The final prediction and summary
        """
        prompt= self.prep_prompt_summarize(analysis_text)

        return self._call_llm(prompt)

 
    
    def _call_llm(self, prompt: str) -> str:
        """
        Helper method to call the LLM with the given prompt.
        
        Args:
            prompt (str): Input prompt for the LLM.
            
        Returns:
            str: LLM's response.
        """
        try:
            # Generate text
            inputs1 = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_ids1 = inputs1["input_ids"]

            with torch.inference_mode():
                outputs1 = self.model.generate(
                    **inputs1,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.9,
                    top_p=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Get only the newly generated tokens (after the input prompt)
            new_tokens1 = outputs1[0][input_ids1.shape[1]:]

            # Decode only the new tokens
            response = self.tokenizer.decode(new_tokens1, skip_special_tokens=True)

            return response
        except Exception as e:
            raise Exception(f"Error calling LLM: {e}")




# # Example usage
# if __name__ == "__main__":
#     # Initialize the interpreter
#     interpreter = TextInterpreter()
    
#     # Example input text and SHAP values
#     text = """
#     The patient is a 65-year-old male who reports increasing forgetfulness over the past year. 
#     He often struggles to recall names of close friends and frequently misplaces items like his keys. 
#     His speech is somewhat halting, with occasional word-finding difficulties.
#     """
    
#     shap_values = {
#         "forgetfulness": 0.8,
#         "recall names": 0.7,
#         "misplaces items": 0.6,
#         "speech is somewhat halting": 0.5,
#         "word-finding difficulties": 0.9
#     }
    
    
#     # Step 1: Initial analysis
#     print("Performing initial analysis...")
#     initial_response = interpreter.analyze_cognitive_impairment(
#         text=text,
#         shap_values=shap_values,
#     )
#     print("\nInitial Analysis Results:")
#     print(initial_response)
    
#     # Example refinement criteria (replace with your actual criteria)
#     refinement_criteria ={"Lexical Richness":0.2}
    
#     # Step 2: Refine the analysis
#     print("\nRefining analysis...")
#     refined_response = interpreter.refine_analysis(
#         previous_response=initial_response,
#         refinement_criteria=refinement_criteria
#     )
#     print("\nRefined Analysis Results:")
#     print(refined_response)