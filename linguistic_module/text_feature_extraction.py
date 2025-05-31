from typing import Dict, List, Optional
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import spacy
import en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer #convert text comment into a numeric vector

import pandas as pd
from copy import deepcopy
import numpy as np
from lexicalrichness import LexicalRichness
from string import punctuation
import math

class TextFeatureExtractor:
    """A modular class for extracting linguistic features from text."""
    
    def __init__(self, text: str):
        """
        Initialize with the input text.
        
        Args:
            text (str): Input text to analyze.
        """

        self.sia = SentimentIntensityAnalyzer()
        self.nlp = en_core_web_sm.load()
        self.pos_vectorizer = CountVectorizer()
        self.tag_vectorizer = CountVectorizer()
        self.content_pos_types = {'ADJ', 'ADV', 'VERB', 'NOUN'}
        self.function_pos_types = {'DET', 'AUX', 'ADP', 'CCONJ', 'PART', 'PRON', 'SCONJ'}
        self.pos_tags = {
            'standard': ["ADJ", "ADV", "DET", "CONJ", "INTJ", "NOUN", "NUM", "IN", "PRON", "VERB"],
            'special_cases': {
                'CONJ': ["CCONJ", "SCONJ"]  # CONJ combines these tags
            }
        }

        self.df = pd.DataFrame({"transcription":text})
        
       

    # ----------------------
    # Core Text Processing Utilities
    # ----------------------
    def preprocess(self, df_text: str) -> str:
        # remove punctuation marks
        df_text['clean_text'] = df_text['transcription'].apply(lambda x: re.sub(r'http\S+', '', str(x)))

        punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

        df_text['clean_text'] = df_text['clean_text'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
        # test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

        # convert text to lowercase
        df_text['clean_text'] = df_text['clean_text'].str.lower()
        # test['clean_tweet'] = test['clean_tweet'].str.lower()

        # remove numbers
        df_text['clean_text'] = df_text['clean_text'].str.replace("[0-9]", " ")
        # test['clean_tweet'] = test['clean_tweet'].str.replace("[0-9]", " ")

        # remove whitespaces
        df_text['clean_text'] = df_text['clean_text'].apply(lambda x:' '.join(x.split()))
        return df_text
    
    def add_pos_features(self, df: pd.DataFrame, text_column: str = "clean_text") -> pd.DataFrame:
        """
        Add Part-of-Speech (POS) features to dataframe
        
        Args:
            df: Input dataframe containing text data
            text_column: Name of column containing clean text
            
        Returns:
            DataFrame with original data plus POS features
        """
        # Generate POS tags
        df["POS"] = df[text_column].apply(self._get_pos_tags)
        
        # Vectorize POS tags
        X_v = self.pos_vectorizer.fit_transform(df["POS"])
        pos_df = pd.DataFrame(
            data=X_v.toarray(),
            columns=[pos.upper() for pos in self.pos_vectorizer.get_feature_names_out()],
            index=df.index
        )
        
        # Combine with original dataframe
        return pd.concat([df, pos_df], axis=1)
    
    def _get_pos_tags(self, text: str) -> str:
        """
        Helper method to generate POS tags string
        
        Args:
            text: Input text to analyze
            
        Returns:
            String of space-separated POS tags
        """
        doc = self.nlp(text)
        return " ".join([word.pos_ for word in doc])
    
    def add_tag_features(self, df: pd.DataFrame, text_column: str = "clean_text") -> pd.DataFrame:
        """
        Add detailed TAG features to dataframe
        
        Args:
            df: Input dataframe containing text data
            text_column: Name of column containing clean text
            
        Returns:
            DataFrame with original data plus TAG features
        """
        # Generate TAG annotations
        df["TAG"] = df[text_column].apply(self._get_tag_annotations)
        
        # Vectorize TAG annotations
        X_tag = self.tag_vectorizer.fit_transform(df["TAG"])
        tag_df = pd.DataFrame(
            data=X_tag.toarray(),
            columns=[f"TAG_{tag.upper()}" for tag in self.tag_vectorizer.get_feature_names_out()],
            index=df.index
        )
        
        # Combine with original dataframe
        return pd.concat([df, tag_df], axis=1)
    
    def _get_tag_annotations(self, text: str) -> str:
        """
        Helper method to generate detailed TAG annotations string
        
        Args:
            text: Input text to analyze
            
        Returns:
            String of space-separated TAG annotations
        """
        doc = self.nlp(text)
        return " ".join([word.tag_ for word in doc])

    # ----------------------
    # Feature Extraction Modules
    # ----------------------
    def calculate_content_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate content density ratio (content words vs function words)
        
        Args:
            df: DataFrame containing POS tag counts
            
        Returns:
            DataFrame with added 'Content_Density' column
            
        Formula:
            Content Density = (Content Words) / (Function Words)
            Where Content Words = ADJ + ADV + VERB + NOUN
            And Function Words = All other non-punctuation POS tags
        """
        # Calculate numerator (content words)
        content_words = sum(df[pos] for pos in self.content_pos_types if pos in df.columns)
        
        # Calculate denominator (function words)
        function_words_cols = [col for col in df.columns 
                             if col in self.function_pos_types and col not in self.content_pos_types]
        function_words = df[function_words_cols].sum(axis=1)
        
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            df['Content_Density'] = np.where(
                function_words > 0,
                content_words / function_words,
                np.nan  # Return NaN when no function words exist
            )
        
        return df

    
    def calculate_pos_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate normalized Part-of-Speech rate across all documents
        
        Args:
            df: DataFrame containing POS tag counts
            
        Returns:
            DataFrame with added 'Part_of_Speech_rate' column
            
        Formula:
            1. For each POS tag, calculate normalized frequency (tag count / total count for that tag)
            2. For special cases (like CONJ), combine specified tags
            3. Average all normalized frequencies
        """
        normalized_rates = []
        
        for tag in self.pos_tags['standard']:
            if tag in self.pos_tags['special_cases']:
                # Handle special combined tags
                combined_tag = sum(df[sub_tag] for sub_tag in self.pos_tags['special_cases'][tag] 
                                 if sub_tag in df.columns)
                tag_rate = combined_tag / (combined_tag.sum() + 1e-10)  # Avoid division by zero
            else:
                # Handle standard tags
                if tag in df.columns:
                    tag_rate = df[tag] / (df[tag].sum() + 1e-10)
                else:
                    tag_rate = pd.Series(0, index=df.index)
            
            normalized_rates.append(tag_rate)
        
        # Calculate average rate across all POS tags
        df['Part_of_Speech_rate'] = sum(normalized_rates) / len(self.pos_tags['standard'])
        
        return df


    def calculate_reality_reference_rate(self, df: pd.DataFrame, 
                                        noun_col: str = "NOUN", 
                                        verb_col: str = "VERB") -> pd.DataFrame:
            """
            Calculate the noun-to-verb ratio (Reference Rate to Reality)
            
            Args:
                df: DataFrame containing POS tag counts
                noun_col: Column name for noun counts (default: "NOUN")
                verb_col: Column name for verb counts (default: "VERB")
                
            Returns:
                DataFrame with added 'Reference_Rate_to_Reality' column
                
            Formula:
                Reference Rate to Reality = NOUN count / VERB count
                - Handles division by zero by returning NaN
                - Adds small epsilon (1e-10) to prevent extreme values
            """
            # Verify required columns exist
            if noun_col not in df.columns or verb_col not in df.columns:
                raise ValueError(f"DataFrame must contain '{noun_col}' and '{verb_col}' columns")
            
            # Calculate ratio with numerical stability
            with np.errstate(divide='ignore', invalid='ignore'):
                df['Reference_Rate_to_Reality'] = np.where(
                    df[verb_col] > 0,
                    df[noun_col] / (df[verb_col] + 1e-10),  # Small epsilon to avoid division by zero
                    np.nan  # Return NaN when no verbs exist
                )
            
            return df

    def calculate_relative_pronoun_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate normalized relative pronoun usage rate across documents
        
        Args:
            df: DataFrame containing POS tag counts (must include WDT and WP columns)
            
        Returns:
            DataFrame with added 'Relative_pronouns_rate' column
            
        Formula:
            Relative_pronouns_rate = (WDT + WP) / sum(WDT + WP across all documents)
            - Returns 0 if no relative pronouns exist in the corpus
            - Handles missing columns gracefully
        """
        # Check for required columns
        required_cols = {'WDT', 'WP'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for relative pronouns: {missing_cols}")
        
        # Calculate relative pronoun frequencies
        relative_pronouns = df['WDT'] + df['WP']
        total_relative_pronouns = relative_pronouns.sum()
        
        # Normalize (handle zero total case)
        df['Relative_pronouns_rate'] = (
            relative_pronouns / (total_relative_pronouns + 1e-10)  # Add epsilon to avoid division by zero
        ) if total_relative_pronouns > 0 else 0.0
        
        return df
    
    def calculate_negative_adverb_rate(self, df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
        """
        Calculate normalized negative adverb usage rate
        
        Args:
            df: DataFrame containing clean text
            text_col: Name of column containing preprocessed text
            
        Returns:
            DataFrame with added 'Negative_adverbs_rate' column
            
        Process:
            1. Identifies adverbs in text
            2. Checks if adverb has negative sentiment
            3. Normalizes counts by total negative adverbs in corpus
        """
        if text_col not in df.columns:
            raise ValueError(f"DataFrame missing required text column: {text_col}")
        
        # Vectorized counting of negative adverbs
        neg_adv_counts = df[text_col].apply(self._count_negative_adverbs)
        total_neg_adv = neg_adv_counts.sum()
        
        # Calculate normalized rate (handle zero case)
        df['Negative_adverbs_rate'] = (
            neg_adv_counts / (total_neg_adv + 1e-10)  # Avoid division by zero
        )
        
        return df
    
    def _count_negative_adverbs(self, text: str) -> int:
        """
        Helper method to count negative adverbs in a single text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Count of negative sentiment adverbs
        """
        doc = self.nlp(text)
        count = 0
        
        for token in doc:
            if token.pos_ == "ADV":
                sentiment = self.sia.polarity_scores(token.text)
                if sentiment['neg'] > sentiment['pos']:
                    count += 1
        return count
    
    def analyze_speech_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze various speech features including filler words, lexical frequency,
        speech rate, and filler rate.
        
        Args:
            df: DataFrame containing clean text and POS tags
            
        Returns:
            DataFrame with added speech feature columns
        """
        # Define filler words list
        filler_list = ["uh", "um", "hmm", "mhm", "huh"]
        
        # Count filler words
        df["Filler words"] = df["clean_text"].apply(
            lambda x: sum(1 for word in self.nlp(x) if str(word.text) in filler_list)
        )
        
        # Calculate lexical frequency
        real_pos_list = ['VERB', 'PROPN', 'NOUN', 'ADV', 'ADJ']
        df["Lexical frequency"] = np.log2(
            df[real_pos_list].sum(axis=1)
        )
        
        # Calculate speech rate (excluding PUNCT)
        pos_columns = df.loc[:, "ADJ":"VERB"].columns
        pos_columns = pos_columns.drop("PUNCT") if "PUNCT" in pos_columns else pos_columns
        df["Speech rate"] = (
            df[pos_columns].sum(axis=1) / df["length_audio_file"]
        )
        
        # Calculate filler rate
        df["Filler rate"] = (
            df["Filler words"] / df["length_audio_file"]
        )
        
        return df
    
    def calculate_grammatical_ratios(self, df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
        """
        Calculate various grammatical ratios relative to total content words
        
        Args:
            df: DataFrame containing text and POS tag counts
            text_col: Name of column containing clean text
            
        Returns:
            DataFrame with added ratio columns:
            - Definite_articles: 'the' count / content words
            - Indefinite_articles: 'a/an' count / content words  
            - Pronouns: PRON / content words
            - Nouns: NOUN / content words
            - Verbs: VERB / content words
            - Determiners: DET / content words
            
        Content words defined as sum of POS tags from ADJ to VERB excluding PUNCT
        """
        # Validate required columns
        required_pos_tags = {'ADJ', 'ADV', 'DET', 'NOUN', 'PRON', 'VERB'}
        missing_tags = required_pos_tags - set(df.columns)
        if missing_tags:
            raise ValueError(f"Missing required POS tag columns: {missing_tags}")
        
        if text_col not in df.columns:
            raise ValueError(f"Missing text column: {text_col}")

        # Calculate denominator (content words)
        content_cols = [col for col in df.loc[:, "ADJ":"VERB"].columns 
                      if col != "PUNCT" and col in df.columns]
        df['_content_words'] = df[content_cols].sum(axis=1)
        
        # Avoid division by zero
        denominator = df['_content_words'].replace(0, np.nan)

        # Calculate definite articles ('the')
        df['Definite_articles'] = df[text_col].apply(
            lambda x: sum(1 for word in self.nlp(x) if str(word.text.lower()) == "the")
        ) / denominator

        # Calculate indefinite articles ('a/an')
        indefinite_articles = {'a', 'an'}
        df['Indefinite_articles'] = df[text_col].apply(
            lambda x: sum(1 for word in self.nlp(x) if str(word.text.lower()) in indefinite_articles)
        ) / denominator

        # Calculate POS tag ratios
        pos_ratios = {
            'Pronouns': 'PRON',
            'Nouns': 'NOUN', 
            'Verbs': 'VERB',
            'Determiners': 'DET'
        }
        
        for new_col, pos_tag in pos_ratios.items():
            if pos_tag in df.columns:
                df[new_col] = df[pos_tag] / denominator

        # Clean up temporary column
        df.drop('_content_words', axis=1, inplace=True)
        
        return df
    
    def analyze_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze content-related speech features including content words and repeated clauses.
        
        Args:
            df: DataFrame containing clean text and POS tags
            
        Returns:
            DataFrame with added content feature columns
        """
        # Calculate content words (non-stop words) ratio
        df["Content words"] = (
            df["clean_text"].apply(
                lambda x: sum(1 for word in self.nlp(x) if not word.is_stop)
            ) / df.loc[:, "ADJ":"VERB"]
            .drop("PUNCT", axis=1, errors='ignore')
            .sum(axis=1)
        )
        
        # Calculate consecutive repeated clauses
        def _find_repeated_clauses(text: str) -> int:
            """Helper function to count repeated clauses in text"""
            matches = re.findall(r'((\b.+?\b)(?:\s\2)+)', text)
            return sum(
                int((len(match[0]) + 1) / (len(match[1]) + 1))
                for match in matches
            )
        
        df["Consecutive repeated clauses"] = (
            df['clean_text'].apply(_find_repeated_clauses)
        )
        
        return df
    
    def calculate_lexical_richness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various lexical richness metrics for text analysis.
        
        Metrics included:
        - Type-Token Ratio (TTR)
        - Root Type-Token Ratio (RTTR)
        - Corrected Type-Token Ratio (CTTR)
        - Mean Segmental Type-Token Ratio (MSTTR)
        - Moving Average Type-Token Ratio (MATTR)
        
        Args:
            df: DataFrame containing clean text
            
        Returns:
            DataFrame with added lexical richness columns
        """
        # Calculate all lexical richness metrics in a single pass to avoid repeated processing
        def _get_lexical_metrics(text: str) -> dict:
            """Helper function to compute all lexical metrics at once"""
            lr = LexicalRichness(text)
            return {
                'TTR': lr.ttr,
                'RTTR': lr.rttr,
                'CTTR': lr.cttr,
                'MSTTR': lr.msttr(segment_size=25),  # Using default segment size of 25
                'MATTR': lr.mattr(window_size=25)    # Using default window size of 25
            }
        
        # Apply the function once and expand results into multiple columns
        lexical_metrics = df['clean_text'].apply(_get_lexical_metrics).apply(pd.Series)
        
        # Add the metrics to the original dataframe
        df = pd.concat([df, lexical_metrics], axis=1)
        
        return df
    
    def calculate_word_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate word count metrics including:
        - Total word count
        - Unique word count
        - Ratio of unique words to total words
        
        Args:
            df: DataFrame containing clean text
            
        Returns:
            DataFrame with added word count columns
        """
        # Calculate all word count metrics in a single pass
        def _get_word_counts(text: str) -> dict:
            """Helper function to compute word counts once"""
            lr = LexicalRichness(text)
            return {
                'Word count': lr.words,
                'Unique Word count': lr.terms,
                'Ratio unique word count to total word count': lr.terms / lr.words if lr.words > 0 else 0
            }
        
        # Apply the function and expand results
        word_counts = df['clean_text'].apply(_get_word_counts).apply(pd.Series)
        
        # Add to original dataframe
        df = pd.concat([df, word_counts], axis=1)
        
        return df
    
    def calculate_advanced_lexical_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced lexical diversity metrics including:
        - Brunet's Index
        - Honoré's Statistic
        - Measure of Textual Lexical Diversity (MTLD)
        - Hypergeometric Distribution Diversity (HDD)
        
        Args:
            df: DataFrame containing clean text and word counts
            
        Returns:
            DataFrame with added advanced lexical metrics columns
        """
        # Calculate Brunet's Index
        df['Brunet\'s Index'] = (
            df['Word count'] ** (df['Unique Word count'] ** (-0.165)))
        
        # Calculate Honoré's Statistic with improved implementation
        def _calculate_honore(text: str) -> float:
            """Calculate Honoré's Statistic with robust error handling"""
            try:
                words = [word.strip(punctuation) for word in text.lower().split() if word.strip(punctuation)]
                N = len(words)
                if N == 0:
                    return np.nan
                    
                types = set(words)
                V = len(types)
                V1 = sum(1 for w in types if words.count(w) == 1)
                
                if V == 0 or V1 == V:  # Prevent division by zero and log(0)
                    return np.nan
                    
                return 100 * math.log(N) / (1 - V1/V)
            except (ValueError, ZeroDivisionError):
                return np.nan
        
        df['Honoré\'s Statistic'] = df['clean_text'].apply(_calculate_honore)
        
        # Fill NA with max value (consider alternative imputation methods)
        max_honore = df['Honoré\'s Statistic'].max(skipna=True)
        df['Honoré\'s Statistic'] = df['Honoré\'s Statistic'].fillna(max_honore)
        
        # Calculate MTLD and HDD in single pass
        def _get_lexical_diversity(text: str):
            """Helper function to compute MTLD and HDD together"""
            lr = LexicalRichness(text)
            return {
                'Measure of Textual Lexical Diversity': lr.mtld(threshold=0.72),
                'Hypergeometric Distribution Diversity': lr.hdd(draws=42)  # Standard 42 draws
            }
        
        lexical_diversity = df['clean_text'].apply(_get_lexical_diversity).apply(pd.Series)
        df = pd.concat([df, lexical_diversity], axis=1)
        
        return df
    # ----------------------
    # Master Feature Extractor
    # ----------------------
    def extract_all_features(self) -> Dict[str, float]:
        """
        Run all feature extractors and return {feature_name: value}.
        Add new features to this dictionary!

        """
        self.df = self.preprocess(deepcopy(self.df))
        self.df = self.add_pos_features(deepcopy(self.df))
        self.df = self.add_tag_features(deepcopy(self.df))
        self.df = self.calculate_content_density(deepcopy(self.df))
        self.df = self.calculate_pos_rate(deepcopy(self.df))
        self.df = self.calculate_reality_reference_rate(deepcopy(self.df))
        self.df = self.calculate_relative_pronoun_rate(deepcopy(self.df))
        self.df = self.calculate_negative_adverb_rate(deepcopy(self.df))
        self.df = self.analyze_speech_features(deepcopy(self.df))
        self.df = self.calculate_grammatical_ratios(deepcopy(self.df))
        self.df = self.analyze_content_features(deepcopy(self.df))
        self.df = self.calculate_lexical_richness(deepcopy(self.df))
        self.df = self.calculate_word_counts(deepcopy(self.df))
        self.df = self.calculate_advanced_lexical_metrics(deepcopy(self.df))

        return self.df.to_dict()


# Example Usage
# if __name__ == "__main__":
#     sample_text = "The cat sat on the mat. The dog barked loudly."
#     extractor = TextFeatureExtractor(sample_text)
    
#     # Extract all features (will raise errors until you implement them!)
#     try:
#         features = extractor.extract_all_features()
#         print(features)
#     except NotImplementedError as e:
#         print(f"Feature not implemented yet: {e}")