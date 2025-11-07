from typing import Dict, List, Union,Optional,Tuple
from modelscope import AutoTokenizer
from modelscope import AutoModelForCausalLM
import torch
import numpy as np
import json
from openai import OpenAI
import openai

system_prompt4 = """
    You are an expert language model designed to detect and interpret linguistic cues indicative of cognitive status from text. You will receive:
    1) A text passage to analyze (transcription of a speaker describing a visual scene, such as the Cookie Theft picture).
    2) A machine learning model’s prediction (healthy or cognitive impairment) and its confidence of that prediction.
    3) Two separate expert interpretations of the text, based on six linguistic categories
    (Lexical Richness, Syntactic Complexity, Disfluencies and Repetition, Semantic Coherence, Difficulty with Spatial Reasoning and Visualization, and Impaired Executive Function):
    - The first one provides a qualitative interpretation across all six categories.
    - The second one includes detailed quantitative measurements for the first four categories.

    Your task is to:
    - Carefully read both expert interpretations.
    - Compare and evaluate them critically. If there are any contradictions, use logical reasoning to resolve them—prioritizing the interpretation that is more strongly supported by the textual evidence and aligns more consistently with known linguistic markers of cognitive status.
    - Synthesize both sources into a unified, coherent analysis that explains how the linguistic features of the passage may reflect healthy or impaired cognition.

    Your output must be structured as **bullet points**, each describing one key aspect of the analysis relevant to cognitive status.

    ---
    ## Text to Analyze:
    {text}
    ---
    ## Model's Prediction / Confidence:
    {model_pred} / {model_conf}
    ---
    ## First Interpretation:
    {shap_interpretation}

    ---
    ## Second Interpretation:
    {feature_interpretation}

    ---
    ## Analysis:
    """

system_prompt3 = """
   You are a specialized language model trained to detect linguistic cues of cognitive status. You will receive:
    1) A detailed explanation of some linguistic features, grouped into four main categories, and their relevance to cognitive status.
    2) A text passage to analyze (transcription of a speaker describing a visual scene, such as the Cookie Theft picture).
    3) A machine learning model’s prediction (healthy or cognitive impairment) and its confidence of that prediction.
    4) The linguistic features values calculated from the text.

    You must analyze the given text and the Linguistic Features and briefly describe the text in terms of the provided linguistic features.
    Use logical reasoning to explain how these features contribute (or do not contribute) to the model’s prediction, supported by values of the relevant linguistic features.
    Keep your output concise, well-supported, insightful, and relevant to cognitive assessment, using bullet points, with each point describing one key aspect of the analysis.

    ---

    ## Detailed Explanation Linguistic Features:
    • Lexical Richness: Reduced vocabulary diversity may reflect word-finding difficulties and lexical retrieval deficits.
        •• Type-Token Ratio (TTR): (0-1) LOW: Repetitive; HIGH: diverse 
        •• Root Type-Token Ratio (RTTR): (2.0-8.0 (Guiraud's Index)) LOW: simple vocab; HIGH: varied vocab
        •• Corrected Type-Token Ratio (CTTR): (1.5-5.0 (Carroll's CTTR)) LOW: restricted vocab; HIGH: rich vocab
        •• Brunet's Index: (~10-100) LOW: diverse; HIGH: limited vocab
        •• Honoré's Statistic: (~0-2000) LOW: low richness; HIGH: high richness
        •• Measure of Textual Lexical Diversity (MTLD): (~10-150) LOW: limited vocab; HIGH: stable diversity
        •• Hypergeometric Distribution Diversity (HDD): (0-1) LOW: low diversity; HIGH: diverse vocab
        •• Ratio unique word count to total word count: (0-1) LOW: repetition; HIGH: variety
        •• Unique Word count: (10-∞) LOW: restricted vocab; HIGH: lexical richness
        •• Lexical frequency: (0-∞) LOW: rare words; HIGH: frequent/common words
        •• Content words ratio: (0-1) LOW: vague; HIGH: info-rich

    • Syntactic Complexity: Simplified grammar and reduced structural variety may signal cognitive decline affecting sentence planning.
        •• Part_of_Speech_rate: (0-1) LOW: reduced variation; HIGH: balanced grammar
        •• Relative_pronouns_rate: (0-1) LOW: simple syntax; HIGH: complex clauses
        •• Determiners Ratio: (0-1) LOW: vague; HIGH: clear reference
        •• Verbs Ratio: (0-1) LOW: static speech; HIGH: dynamic structure
        •• Nouns Ratio: (0-1) LOW: low content; HIGH: info-dense
        •• Negative_adverbs_rate: (0-1) LOW: less negation; HIGH: complex expression
        •• Word count: (10-∞) LOW: brevity; HIGH: verbosity/planning

    • Disfluencies and Repetition: Frequent hesitations, fillers, or repeated phrases may reflect planning difficulties and reduced cognitive flexibility.
        •• Speech rate (wps): (2.3-3.3 wps) LOW: slowed cognition; HIGH: normal/pressured
        •• Consecutive repeated clauses count: (0-∞) LOW: flexible; HIGH: perseveration

    • Semantic Coherence and Referential Clarity: Vague references and reduced cohesion may indicate impaired semantic organization and discourse tracking.
        •• Content_Density: (0-1) LOW: vague; HIGH: info-rich
        •• Reference_Rate_to_Reality: (0-∞) LOW: abstract; HIGH: concrete info
        •• Pronouns Ratio: (0-1) LOW: specific; HIGH: ambiguous
        •• Definite_articles Ratio: (0-1) LOW: vague; HIGH: specific reference
        •• Indefinite_articles Ratio: (0-1) LOW: specific; HIGH: general

    ---
    ## Text to Analyze:
    {text}
    ---

    ## Model's Prediction / Confidence:
    {model_pred} / {model_conf}
    ---

    ## Linguistic Features Values:
    {linguistic_features}

    ---
    ## Analysis:

"""
system_prompt1 = """
    You are a specialized language model trained to detect linguistic cues of cognitive status. You will receive:
    1) A set of linguistic features to consider.
    2) A text passage to analyze (transcription of a speaker describing a visual scene, such as the Cookie Theft picture).
    3) A machine learning model’s prediction (healthy or cognitive impairment) and its confidence of that prediction.

    You must analyze the given text and briefly describe the text in terms of the provided linguistic features.
    Use logical reasoning to explain how these features contribute (or do not contribute) to the model’s prediction.
    Keep your output concise, well-supported, insightful, and relevant to cognitive assessment, using bullet points, with each point describing one key aspect of the analysis.

    ---
    ## Linguistic Features to Consider:
    • Lexical Richness: Vocabulary diversity, word-finding issues, or overuse of vague words.E.g., using “thing” or “stuff” instead of specific nouns.
    • Syntactic Complexity: Simplified grammar, limited sentence structure, or grammatical errors.
    • Disfluencies and Repetition: Frequent pauses, fillers (e.g., “um,” “uh”), or repeated words.
    • Semantic Coherence: Vague references, disorganized ideas, or unclear meaning. E.g., “They’re doing something over there with it.”
    • Difficulty with Spatial Reasoning and Visualization: Trouble describing where things are. E.g., “It’s next to... no, behind... or maybe in front of it.”
    • Impaired Executive Function: Disorganized or off-topic speech, poor sequencing. E.g., Jumps between unrelated actions or events without completing ideas.
    • Additional Feature: Include any other relevant linguistic feature that may be indicative of cognitive status.
    ---

    ## Text to Analyze:
    {text}
    ---
    ## Model's Prediction / Confidence:
    {model_pred} / {model_conf}
    ---
    ## Analysis:

    """

system_prompt2 = """
        You are an expert language model designed to detect and interpret linguistic cues indicative of cognitive status from text. You will receive:
        1) A text passage (transcription of a speaker describing a visual scene, such as the Cookie Theft picture).
        2) A detailed analysis of the text across six linguistic categories (Lexical Richness, Syntactic Complexity, Disfluencies and Repetition, Semantic Coherence, Difficulty with Spatial Reasoning and Visualization, and Impaired Executive Function).
        3) A machine learning model’s prediction (healthy or cognitively impaired) and its confidence score.

        Your task is to:
        - Read the analysis carefully.
        - Identify the four linguistic categories that most strongly support the model’s prediction.
        - Summarize these four categories and their implications in **bullet points**, beginning each bullet with **the exact name** of the six linguistic categories (Lexical Richness, Syntactic Complexity, Disfluencies and Repetition, Semantic Coherence, Difficulty with Spatial Reasoning and Visualization, and Impaired Executive Function).
        - For each bullet point, include specific **examples or evidence from the text** to support it.
        - Write the final prediction (healthy or cognitively impaired) in **one short sentence** without using bullet point.

        **Do not add extra explanations or points.**
        **Avoid referencing any quantitative measurements from the text or offering suggestions for further analysis.**
        **Do not repeat yourself.**
        ---
        ## Text Passage:
        {text}
        ---
        ## Long Analysis:
        {analysis_text}
        ---
        ## Model's Prediction / Confidence:
        {model_pred} / {model_conf}
        ---
        ## Key findings:

        """
class TextInterpreter:
    """
    A class to interpret SHAP values and analyze text for cognitive impairment cues using an LLM.
    """

    def __init__(self, model_path: Optional[str] = None, openai_config: Optional[Dict] = None):
        """
        Initialize the interpreter with either a local model or OpenAI API.

        Args:
            model_path: Path to local HuggingFace model. If None, uses OpenAI.
            openai_config: Dictionary with 'api_key' and 'base_url' for OpenAI.

        Raises:
            ValueError: If neither model_path nor openai_config is provided
        """
        if not model_path and not openai_config:
            raise ValueError("Either model_path or openai_config must be provided")

        self.model, self.tokenizer = self._initialize_model(model_path, openai_config)
        self.label_mapping = {0: "healthy", 1: "cognitive impairment", 2: "cognitive impairment"}

    def _initialize_model(self,
                         model_path: Optional[str],
                         openai_config: Optional[Dict]) -> Tuple[Union[AutoModelForCausalLM, OpenAI], Optional[AutoTokenizer]]:
        """
        Initializes and returns the language model and tokenizer.

        Args:
            model_path: Path to local HuggingFace model
            openai_config: Configuration for OpenAI API

        Returns:
            Tuple of (model, tokenizer). Tokenizer is None for OpenAI.
        """
        if model_path:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            return model, tokenizer

        # OpenAI case
        if not openai_config.get('api_key'):
            raise ValueError("OpenAI config must include 'api_key'")

        model = OpenAI(
            api_key=openai_config['api_key'],
            base_url=openai_config.get('base_url')
        )
        return model, None

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
            shap_value = str(shap_value) if shap_value > 0 else ''
            token_value_pairs.append((token_str, shap_value))

        return token_value_pairs

    def SHAP_values_interpretation(self, transcription: str, shap_values: Union[Dict, List],shap_index:int,model_conf:float) -> str:
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

        # prompt = system_prompt1.format(text=transcription, shap_values=json.dumps(token_shap_pairs, indent=2),model_pred=self.label_mapping[shap_index],model_conf=model_conf)
        prompt = system_prompt1.format(text=transcription,model_pred=self.label_mapping[shap_index],model_conf=model_conf)

        # Call the LLM
        response = self._call_llm(prompt)
        return prompt, response

    def combine_interpretation(self, transcription: str,shap_interpretation: str,feature_interpretation: str,model_pred:int,model_conf:float) -> str:

        prompt = system_prompt4.format(text=transcription, shap_interpretation=shap_interpretation, feature_interpretation=feature_interpretation,model_pred=self.label_mapping[model_pred],model_conf=model_conf)

        # Call the LLM
        response = self._call_llm(prompt)
        return prompt, response

    def linguistic_features_interpretation(self, transcription: str,linguistic_features: Dict[str, float],model_pred:int,model_conf:float ) -> str:
        """
        Generates the linguistic analysis using the provided transcription and linguistic features.

        Args:
            transcription (str): Text to analyze
            linguistic features: linguistic features for interpretation

        Returns:
            str: The generated analysis text
        """
        features = "\n".join(
            [f"- {metric.replace('_', ' ').title()}: {value}"
            for metric, value in linguistic_features.items()]
        )
        prompt = system_prompt3.format(text=transcription, linguistic_features=features,model_pred=self.label_mapping[model_pred],model_conf=model_conf)

        # Call the LLM
        response = self._call_llm(prompt)
        return prompt, response

    def prep_prompt_summarize(self,generated_text):
        content = system_prompt2.format(generated_text=generated_text)
        messages = [{"role": "user", "content": content}]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return prompt

    def generate_final_interpretation(self,transcription:str,analysis_text: str,model_pred:int,model_conf:float) -> str:
        """
        Generates a final interpretation based on the previous analysis of text.

        Args:
            analysis_text (str): The initial analysis to summarize

        Returns:
            str: The final prediction and summary
        """
        # prompt= self.prep_prompt_summarize(analysis_text)
        prompt= system_prompt2.format(text=transcription , analysis_text=analysis_text,model_pred=self.label_mapping[model_pred],model_conf=model_conf )

        return prompt,self._call_llm(prompt)

    def get_all_interpretations(self,transcription,predicted_label, shap_values, features, probabilities):
        """
        Runs all interpretation steps and returns their results.

        Args:
            shap_values: SHAP values for interpretation
            features: Linguistic features for interpretation
            probabilities: Model confidence probabilities

        Returns:
            tuple: (prompt1, shap_interp,
                prompt2, ling_interp,
                prompt3, combined_interp,
                prompt4, final_interp)
        """
        # SHAP values interpretation
        prompt1, shap_interp = self.SHAP_values_interpretation(
            transcription=transcription,
            shap_values=shap_values,
            shap_index=predicted_label,
            model_conf=probabilities[predicted_label]
        )

        # Linguistic features interpretation
        prompt2, ling_interp = self.linguistic_features_interpretation(
            transcription=transcription,
            linguistic_features=features,
            model_pred=predicted_label,
            model_conf=probabilities[predicted_label]
        )

        # Combined interpretation
        prompt3, combined_interp = self.combine_interpretation(
            transcription=transcription,
            shap_interpretation=shap_interp,
            feature_interpretation=ling_interp,
            model_pred=predicted_label,
            model_conf=probabilities[predicted_label]
        )

        # Final interpretation
        prompt4, final_interp = self.generate_final_interpretation(
            transcription=transcription,
            analysis_text=combined_interp,
            model_pred=predicted_label,
            model_conf=probabilities[predicted_label]
        )

        return (prompt1, shap_interp,
                prompt2, ling_interp,
                prompt3, combined_interp,
                prompt4, final_interp)


    def _call_llm(self, prompt: str, **generation_params) -> str:
        """
        Generate a response from the LLM with the given prompt.

        Args:
            prompt: Input text for the model
            **generation_params: Additional parameters for text generation
                (max_tokens, temperature, etc.)

        Returns:
            The generated text response

        Raises:
            RuntimeError: If there's an error during generation
        """
        try:
            if self.tokenizer:
                return self._generate_local(prompt, **generation_params)
            return self._generate_openai(prompt, **generation_params)
        except Exception as e:
            raise RuntimeError(f"Text generation failed: {str(e)}") from e

    def _generate_local(self, prompt: str, **kwargs) -> str:
        """Generate text using local HuggingFace model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        generation_params = {
            'max_new_tokens': kwargs.get('max_new_tokens', 512),
            'do_sample': kwargs.get('do_sample', True),
            'temperature': kwargs.get('temperature', 0.9),
            'top_p': kwargs.get('top_p', 1.0),
            'eos_token_id': self.tokenizer.eos_token_id,
            **kwargs
        }

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_params)

        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def _generate_openai(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        response = self.model.chat.completions.create(
            model=kwargs.get('model', "meta-llama/llama-3.1-70b-instruct"),
            
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 2048)
        )
        return response.choices[0].message.content




