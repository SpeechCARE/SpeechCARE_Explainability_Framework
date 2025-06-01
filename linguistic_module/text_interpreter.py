from typing import Dict, List, Union
from modelscope import AutoTokenizer
from modelscope import AutoModelForCausalLM
import torch
import numpy as np
import json

# Define interpretations and directions for each feature
metric_interpretations = {
    'Content_Density': "Higher = more informative speech; Lower = vague or empty content",
    'Part_of_Speech_rate': "Higher = greater syntactic richness; Lower = limited structure use",
    'Reference_Rate_to_Reality': "Higher = more concrete reference; Lower = vague or abstract reference",
    'Relative_pronouns_rate': "Higher = more complex syntax; Lower = simplified sentence structure",
    'Negative_adverbs_rate': "Higher = more negation/adversity expressed; may signal emotional tone or confusion",
    'Filler words': "Higher = more hesitation/disfluency; Lower = fluent delivery",
    'word_count': "Higher = more elaboration or verbosity; Lower = minimal content",
    'Lexical frequency': "Higher = use of common/simple words; Lower = rarer or more diverse vocabulary",
    'Speech rate': "Higher = fluent/pressured speech; Lower = slowed/circumlocutory delivery",
    'Filler rate': "Higher = frequent disfluency; Lower = smooth flow",
    'rate_basis': "Contextual rate baseline; interpret in combination with speech rate",
    'Definite_articles': "Higher = specific reference; overuse may reflect compensation or reduced naming ability",
    'Indefinite_articles': "Higher = vague/general reference; may reflect word-finding issues",
    'Pronouns': "Higher = less specific reference; may indicate reduced lexical access",
    'Nouns': "Higher = more object/subject specificity; Lower = lexical retrieval difficulty",
    'Verbs': "Higher = syntactic action richness; Lower = sentence simplification",
    'Determiners': "Higher = more grammatical structure; Lower = syntactic degradation",
    'Content words': "Higher = richer expression; Lower = vague or incoherent discourse",
    'Consecutive repeated clauses': "Higher = perseveration or reduced working memory; Lower = fluent speech",
    'Type-Token Ratio (TTR)': "Higher = more lexical variety; Lower = repetitive vocabulary",
    'Root Type-Token Ratio (RTTR)': "Higher = more lexical variety (normalized); Lower = reduced richness",
    'Corrected Type-Token Ratio (CTTR)': "Higher = richer vocabulary; Lower = simplified lexicon",
    'Word count': "Higher = elaborated response; Lower = terse or minimal output",
    'Unique Word count': "Higher = lexical diversity; Lower = redundancy or retrieval failure",
    'Ratio unique word count to total word count': "Higher = richer vocabulary; Lower = repetition",
    "Brunet's Index": "Lower = more lexical richness; Higher = limited vocabulary",
    "Honoré's Statistic": "Higher = greater lexical variety; Lower = limited vocabulary",
    'Measure of Textual Lexical Diversity': "Higher = diverse vocabulary; Lower = redundancy",
    'Hypergeometric Distribution Diversity': "Higher = high lexical variation; Lower = limited expression"
}

system_prompt3 = """ 
    Now you will receive a set of linguistic features to consider and you need to refine your previous response accordingly.

      
    ---
    ## Linguistic Features to Consider:
    • Lexical Richness: Captured by Type-Token Ratio (TTR), Root TTR, Corrected TTR, Brunet's Index, Honoré's Statistic, MTLD, HDD, and Ratio of Unique to Total Word Count. Lower values may suggest limited vocabulary or word retrieval deficits common in ADRD.

    • Syntactic Complexity: Informed by Part_of_Speech_rate, Relative_pronouns_rate, Determiners, Verbs, and use of complex forms like Relative Pronouns. Declines may reflect grammatical simplification.

    • Sentence Length and Structure: Approximated by Word count, Unique Word count, and Consecutive repeated clauses. Short or repetitive sentence structures can indicate working memory or planning issues.

    • Repetition: Directly captured by Consecutive repeated clauses. High rates may suggest perseveration or reduced cognitive flexibility.

    • Disfluencies and Fillers: Measured by Filler words and Filler rate. Higher usage may reflect hesitation, planning difficulty, or word-finding trouble.

    • Semantic Coherence and Content Density: Based on Content_Density and Reference_Rate_to_Reality. Lower content density or vague references can indicate impaired semantic organization.

    • Referential Clarity and Pronoun Use: Captured by Pronouns, Definite_articles, Indefinite_articles, and Reference_Rate_to_Reality. Overuse of pronouns or poor reference to specific entities may indicate trouble maintaining discourse cohesion.

    ---

    ## Previous Response:
    {previous_response}

    ----
    ## Linguistic Features:
    {linguistic_features}

    ---
    You must refine your previous response using the linguistic features based on:
    Synthesize the significance of provided features to explain how they collectively point to healthy cognition or potential cognitive impairment.
    Ensure that the explanations are concise, insightful, and relevant to cognitive impairment assessment.
    Output should be structured as **bullet points**, with each bullet clearly describing one key aspect of the analysis. 
    ---
    ## Analysis:
"""
system_prompt1 = """
    You are a specialized language model trained to detect linguistic cues of cognitive impairment. You will receive:
    1) A text passage to analyze.
    2) Token-level SHAP values from a pre-trained model.
  
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
    
    def __init__(self,model_path: str):
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
        shap_values_ = shap_values
        shap_values_.values = shap_values_.values[0,:,shap_index]
        token_shap_pairs = self.format_shap_values(shap_values_) 

        prompt = system_prompt1.format(text=transcription, shap_values=json.dumps(token_shap_pairs, indent=2))
        
        # Call the LLM
        response = self._call_llm(prompt)
        return prompt, response
    
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
       

        # Format the metrics with interpretation
        metrics_description = "\n".join(
            [f"- {metric.replace('_', ' ').title()}: {value} → {metric_interpretations.get(metric, 'No interpretation available')}"
            for metric, value in refinement_metrics.items()]
        )

        
        input_message = system_prompt3.format(previous_response=previous_response, linguistic_features=metrics_description)
        
        return input_message, self._call_llm(input_message)
    
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


