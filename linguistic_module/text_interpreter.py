from typing import Dict, List, Union,Optional,Tuple
from modelscope import AutoTokenizer
from modelscope import AutoModelForCausalLM
import torch
import numpy as np
import json
from openai import OpenAI
import openai

system_prompt4 = """
        You are a specialized language model trained to detect linguistic cues of cognitive impairment. You will receive:
        1) A text passage.
        2) Two separate interpretations of that text:
        - One based on token-level SHAP values.
        - One based on predefined linguistic features and their relation to cognitive impairment.

        Your task is to:
        - Carefully read both interpretations.
        - Resolve any contradictions or inconsistencies using logical reasoning.
        - Synthesize the two into a single, coherent explanation focused on how the text may reflect healthy or impaired cognition.
        - If conflicts arise, prioritize the explanation most consistent with known linguistic markers of cognitive decline.

        Your output must be structured as **bullet points**, each describing one key aspect of the combined interpretation relevant to cognitive impairment.

        ---
        ## Text to Analyze:
        {text}

        ---
        ## SHAP-Based Interpretation:
        {shap_interpretation}

        ---
        ## Linguistic Feature-Based Interpretation:
        {feature_interpretation}

        ---
        ## Combined Analysis:
        """

system_prompt3 = """
    You are a specialized language model trained to detect linguistic cues of cognitive status. You will receive:
    1) A text passage to analyze.
    2) A detailed explanation of some linguistic features, grouped into four main categories, and their relevance to cognitive impairment.
    3) The linguistic features values from the text.

    You must analyze the given text and the Linguistic Features based on:
    Synthesize the significance of provided features to explain how they collectively point to healthy cognition or potential cognitive impairment.
    Ensure that the explanations are concise, insightful, and relevant to cognitive impairment assessment.
    Output should be structured as **bullet points**, with each bullet clearly describing one key aspect of the analysis.

    ---
    ## Text to Analyze:
    {text}

    ---
    ## Detailed Explanation Linguistic Features:
    • Lexical Richness: Reduced vocabulary diversity may reflect word-finding difficulties and lexical retrieval deficits common in ADRD.
        •• Type-Token Ratio (TTR): (0-1) LOW: limited vocab; HIGH: diverse vocab
        •• Root Type-Token Ratio (RTTR): (0-1) LOW: simple vocab; HIGH: varied vocab
        •• Corrected Type-Token Ratio (CTTR): (0-1) LOW: restricted vocab; HIGH: rich vocab
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
        •• Filler words: (0-∞) LOW: fluent; HIGH: hesitation
        •• Filler rate: (0-1) LOW: smooth flow; HIGH: planning issues
        •• Speech rate: (wpm) LOW: slowed cognition; HIGH: normal/pressured
        •• Consecutive repeated clauses count: (0-∞) LOW: flexible; HIGH: perseveration

    • Semantic Coherence and Referential Clarity: Vague references and reduced cohesion may indicate impaired semantic organization and discourse tracking.
        •• Content_Density: (0-1) LOW: vague; HIGH: info-rich
        •• Reference_Rate_to_Reality: (0-∞) LOW: abstract; HIGH: concrete info
        •• Pronouns Ratio: (0-1) LOW: specific; HIGH: ambiguous
        •• Definite_articles Ratio: (0-1) LOW: vague; HIGH: specific reference
        •• Indefinite_articles Ratio: (0-1) LOW: specific; HIGH: general


    ----
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
    4) Token-level SHAP values from the model.

    You must analyze the given text and the SHAP values and briefly describe the text in terms of the provided linguistic features.
    Use logical reasoning to explain how these features contribute (or do not contribute) to the model’s prediction, supported by SHAP values, referencing SHAP values when relevant.
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
    ## Token-level SHAP Values:
    {shap_values}
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
            base_url=openai_config.get('base_url'),
            max_tokens= 4096 
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

        prompt = system_prompt1.format(text=transcription, shap_values=json.dumps(token_shap_pairs, indent=2),model_pred=self.label_mapping[shap_index],model_conf=model_conf)

        # Call the LLM
        response = self._call_llm(prompt)
        return prompt, response

    def combine_interpretation(self, transcription: str,shap_interpretation: str,feature_interpretation: str ) -> str:

        prompt = system_prompt4.format(text=transcription, shap_interpretation=shap_interpretation, feature_interpretation=feature_interpretation)

        # Call the LLM
        response = self._call_llm(prompt)
        return prompt, response

    def linguistic_features_interpretation(self, transcription: str,linguistic_features: Dict[str, float]) -> str:
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
        prompt = system_prompt3.format(text=transcription, linguistic_features=features)

        # Call the LLM
        response = self._call_llm(prompt)
        return prompt, response

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
            max_tokens=kwargs.get('max_tokens', 512)
        )
        return response.choices[0].message.content




