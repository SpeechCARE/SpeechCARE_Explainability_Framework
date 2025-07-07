def visualize_linguistic_features(feature_dict):
    """
    Generates a clean HTML table for linguistic features, grouped by category.
    
    Args:
        feature_dict (dict): Dictionary of {feature_name: value}
    
    Returns:
        str: HTML code for visualization
    """

    # Category descriptions and feature metadata
    categories = {
        "Lexical Richness": {
            "description": "Reduced vocabulary diversity may reflect word-finding difficulties and lexical retrieval deficits.",
            "features": {
                "Type-Token Ratio (TTR)": {"range": "0-1", "low": "repetitive", "high": "diverse"},
                "Root Type-Token Ratio (RTTR)": {"range": "2.0-8.0 (Guiraud's Index)", "low": "simple vocab", "high": "varied vocab"},
                "Corrected Type-Token Ratio (CTTR)": {"range": "1.5-5.0 (Carroll's CTTR)", "low": "restricted vocab", "high": "rich vocab"},
                "Brunet's Index": {"range": "~10-100", "low": "diverse", "high": "limited vocab"},
                "Honoré's Statistic": {"range": "~0-2000", "low": "low richness", "high": "high richness"},
                "Measure of Textual Lexical Diversity (MTLD)": {"range": "~10-150", "low": "limited vocab", "high": "stable diversity"},
                "Hypergeometric Distribution Diversity (HDD)": {"range": "0-1", "low": "low diversity", "high": "diverse vocab"},
                "Ratio unique word count to total word count": {"range": "0-1", "low": "repetition", "high": "variety"},
                "Unique Word count": {"range": "10-∞", "low": "restricted vocab", "high": "lexical richness"},
                "Lexical frequency": {"range": "0-∞", "low": "rare words", "high": "frequent/common words"},
                "Content words ratio": {"range": "0-1", "low": "vague", "high": "info-rich"}
            }
        },
        "Syntactic Complexity": {
            "description": "Simplified grammar and reduced structural variety may signal cognitive decline affecting sentence planning.",
            "features": {
                "Part_of_Speech_rate": {"range": "0-1", "low": "reduced variation", "high": "balanced grammar"},
                "Relative_pronouns_rate": {"range": "0-1", "low": "simple syntax", "high": "complex clauses"},
                "Determiners Ratio": {"range": "0-1", "low": "vague", "high": "clear reference"},
                "Verbs Ratio": {"range": "0-1", "low": "static speech", "high": "dynamic structure"},
                "Nouns Ratio": {"range": "0-1", "low": "low content", "high": "info-dense"},
                "Negative_adverbs_rate": {"range": "0-1", "low": "less negation", "high": "complex expression"},
                "Word count": {"range": "10-∞", "low": "brevity", "high": "verbosity/planning"}
            }
        },
        "Disfluencies and Repetition": {
            "description": "Frequent hesitations, fillers, or repeated phrases may reflect planning difficulties and reduced cognitive flexibility.",
            "features": {
                "Speech rate (wps)": {"range": "2.3-3.3 wps", "low": "slowed cognition", "high": "normal/pressured"},
                "Consecutive repeated clauses count": {"range": "0-∞", "low": "flexible", "high": "perseveration"}
            }
        },
        "Semantic Coherence and Referential Clarity": {
            "description": "Vague references and reduced cohesion may indicate impaired semantic organization and discourse tracking.",
            "features": {
                "Content_Density": {"range": "0-1", "low": "vague", "high": "info-rich"},
                "Reference_Rate_to_Reality (noun-to-verb ratio)": {"range": "0-∞", "low": "abstract", "high": "concrete info"},
                "Pronouns Ratio": {"range": "0-1", "low": "specific", "high": "ambiguous"},
                "Definite_articles Ratio": {"range": "0-1", "low": "vague", "high": "specific reference"},
                "Indefinite_articles Ratio": {"range": "0-1", "low": "specific", "high": "general"}
            }
        }
    }

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { 
                font-family: 'Arial', sans-serif; 
                line-height: 1.6; 
                color: #333; 
                max-width: 1000px; 
                margin: 0 auto; 
                padding: 20px;
                background-color: #fafafa;
            }
            .category { 
                background-color: white; 
                border-radius: 8px; 
                padding: 20px; 
                margin-bottom: 25px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                border-left: 4px solid #1E3658;
            }
            .category-title { 
                font-size: 1.3em; 
                font-weight: bold; 
                color: #1E3658;  /* Dark Blue */
                margin-top: 0;
                margin-bottom: 10px;
            }
            .category-desc { 
                margin-bottom: 15px; 
                color: #5a6a7a;
                line-height: 1.5;
            }
            table { 
                width: 100%; 
                border-collapse: separate;
                border-spacing: 0;
                margin-bottom: 15px;
            }
            th { 
                background-color: #1E3658;  /* Dark Blue */
                color: white;
                text-align: left; 
                padding: 12px 10px;
                font-weight: 600;
            }
            td { 
                padding: 10px; 
                border-bottom: 1px solid #e0e0e0; 
                vertical-align: top;
            }
            tr:nth-child(even) { 
                background-color: #f8f8f8; 
            }
            tr:hover {
                background-color: #f0f7f0;  /* Pale Green tint */
            }
            .value { 
                font-weight: bold; 
                color: #1E3658;  /* Dark Blue */
            }
            .range { 
                font-family: 'Courier New', monospace; 
                color: #7FA37F;  /* Pale Green */
                font-weight: 500;
            }
            .interpretation { 
                font-size: 0.9em; 
                color: #6d7a88;
                line-height: 1.4;
            }
            .missing {
                color: #95a5a6; 
                font-style: italic;
            }
        </style>
    </head>
    <body>
    """

    for category, data in categories.items():
        html += f"""
        <div class="category">
            <h2 class="category-title">{category}</h2>
            <p class="category-desc">{data['description']}</p>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th>Normal Range</th>
                    <th>Interpretation</th>
                </tr>
        """

        for feature, meta in data['features'].items():
            value = feature_dict.get(feature, "N/A")
            if value == None:
                value = meta['range'].split('-')[0]
            else:
                try:
                    value = round(float(value),3)
                except:
                    value = value
            interpretation = f"LOW: {meta['low']}; HIGH: {meta['high']}"
            
            html += f"""
                <tr>
                    <td>{feature}</td>
                    <td class="value">{value}</td>
                    <td class="range">{meta['range']}</td>
                    <td class="interpretation">{interpretation}</td>
                </tr>
            """

        html += """
            </table>
        </div>
        """

    html += """
    </body>
    </html>
    """
    
    return html