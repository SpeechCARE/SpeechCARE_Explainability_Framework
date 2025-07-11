import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import Optional, Tuple, Union, Any,List,Dict

def categorize_pauses(num_pauses):
    if num_pauses == 0:
        return 0,"None"
    elif num_pauses == 1:
        return 1,"Single"
    elif num_pauses in [2, 3]:
        return 2,"Few"
    else:
        return 3,"Several"

def categorize_rhythmic_structure(flat_segments: List[Tuple[float, float]]) -> str:
    """
    Categorizes speech rhythm based on flat/monotonous segments in audio.
    
    Args:
        flat_segments: List of (start, end) tuples representing flat segments in seconds.
                       Empty list indicates no flat segments.
    
    Returns:
        One of four rhythm categories:
        - 'Rhythmic': No flat segments
        - 'Relatively Rhythmic': Minimal flat segments (1 segment of 5-6s)
        - 'Less Rhythmic': Some flat segments (2 segments of 5-6s or 1 segment of 6-10s)
        - 'Non-Rhythmic': Significant flat segments (>2 segments or any segment >10s)
    """
    if not flat_segments:
        return 0,"Rhythmic"
    
    durations = [end - start for start, end in flat_segments]
    segment_count = len(durations)
    has_long_segment = any(d > 10 for d in durations)
    has_medium_segment = any(6 < d <= 10 for d in durations)
    all_medium = all(5 <= d <= 6 for d in durations)
    
    if segment_count == 1 and 5 <= durations[0] <= 6:
        return 1,"Relatively Rhythmic"
    elif (segment_count == 2 and all_medium) or has_medium_segment:
        return 2,"Less Rhythmic"
    elif segment_count > 2 or has_long_segment:
        return 3, "Non-Rhythmic"
    return 0,"Rhythmic"


def generate_vocal_analysis_report(
    sample_name: str,
    f0_analysis: Dict[str, Union[str, float, Dict]],  # {'value': float, 'category': str, 'ranges': Dict}
    f3_analysis: Dict[str, Union[str, float, Dict]], 
    pause_count: int,
    flat_segments: List[Tuple[float, float]],
    shimmer_analysis: Dict[str, Union[str, float, Dict]],
    energy_analysis: Dict[str, Union[str, float, Dict]]
) -> str:
    """
    Generate an interactive HTML report for vocal feature analysis with dynamic ranges.
    
    Args:
        sample_name: Name/ID of the audio sample
        f0_analysis: Fundamental frequency analysis results
        f3_analysis: Formant frequency analysis results
        pause_count: Number of noun pauses detected
        flat_segments: List of (start,end) timestamps for flat/monotonous segments
        shimmer_analysis: Shimmer analysis results
        energy_analysis: Energy analysis results
        
    Returns:
        HTML string with dynamic ranges
    """
    # Calculate rhythm metrics
    total_flat_duration = sum(end - start for start, end in flat_segments)
    longest_flat = max([end - start for start, end in flat_segments] or [0])
    rhythm_index , rhythm_category = categorize_rhythmic_structure(flat_segments)
    pause_index , pause_category = categorize_pauses(pause_count)
    shimmer_index = int(shimmer_analysis['category'].split(':')[0][1:]) - 1
    energy_index = int(energy_analysis['category'].split(':')[0][1:]) - 1
    f0_index = int(f0_analysis['category'].split(':')[0][1:]) - 1
    f3_index = int(f3_analysis['category'].split(':')[0][1:]) - 1
    features_index = {
    'pause': pause_index,
    'energy': energy_index,
    'entropy': rhythm_index,
    'shimmer': shimmer_index,
    'f0': f0_index,
    'f3': f3_index
    }
        
    def create_ranges_table(ranges_dict: Dict, feature_type: str) -> str:
        """Generate HTML table rows for value ranges with interpretations
        
        Args:
            ranges_dict: Analysis dictionary containing 'ranges'
            feature_type: Type of feature ('shimmer', 'energy', 'f0', 'f3')
            
        Returns:
            HTML string with table rows
        """
        # Interpretation guides for each feature type
        interpretations = {
            'shimmer': {
                'Stable': 'Normal vocal fold vibration with minimal instability',
                'Almost Stable': 'Mild vocal instability, typically not noticeable',
                'Almost Unstable': 'Moderate vocal instability that may affect voice quality',
                'Unstable': 'Severe vocal instability, often clinically noticable'
            },
            'energy': {
                'Very Low': 'Markedly reduced vocal intensity',
                'Low': 'Below-average vocal intensity',
                'Moderate': 'Typical vocal intensity range',
                'High': 'Elevated vocal intensity'
            },
            'f0': {
                'Very Flat': 'Monotonous speech',
                'Slightly Flat': 'Reduced pitch variation',
                'Natural': 'Natural pitch modulation',
                'Dynamic': 'Strong pitch dynamics, energetic and engaging voice'
            },
            'f3': {
                'Very Limited Coordination': 'Very limited tongue–lip coordination',
                'Limited Coordination': 'Below-average tongue-lip coordination',
                'Normal Coordination': 'Healthy tongu e–lip coordination',
                'High Coordination': 'Healthy dynamic tongue–lip coordination and well-controlled articulation'
            }
        }
        
        rows = []
        for category_name, (start, end) in ranges_dict['ranges'].items():
            # Extract just the descriptive part after "Q1: ", "Q2: " etc.
            display_name = category_name.split(': ')[1] if ': ' in category_name else category_name
            interpretation = interpretations[feature_type].get(display_name, 'No interpretation available')
            
            rows.append(f"""
                <tr>
                    <td>{start:.2f} to {end:.2f}</td>
                    <td>{display_name}</td>
                    <td>{interpretation}</td>
                </tr>
            """)
        return '\n'.join(rows)
    
    def create_pause_table() -> str:
        """Generate HTML table for pause analysis with interpretations"""
        pause_interpretations = {
            'None': 'Normal speech flow without interruptions',
            'Single': 'Minimal pausing, likely natural hesitation',
            'Few': 'Moderate pausing that may affect fluency',
            'Several': 'Excessive pausing, potentially clinically significant'
        }
        
        rows = []
        for count, category in [('0 pauses', 'None'),
                              ('1 pause', 'Single'),
                              ('2-3 pauses', 'Few'),
                              ('>3 pauses', 'Several')]:
            interpretation = pause_interpretations.get(category, '')
            rows.append(f"""
                <tr>
                    <td>{count}</td>
                    <td>{category}</td>
                    <td>{interpretation}</td>
                </tr>
            """)
        return '\n'.join(rows)
    
    def create_rhythm_table() -> str:
        """Generate HTML table for rhythm analysis with interpretations"""
        rhythm_interpretations = {
            'Rhythmic': 'Normal speech rhythm with good prosody',
            'Relatively Rhythmic': 'Mild rhythm deviations, mostly natural',
            'Less Rhythmic': 'Noticeable rhythm disturbances',
            'Non-Rhythmic': 'Severe rhythm disturbance, potentially pathological'
        }
        
        rows = []
        for criteria, category in [('No flat segments', 'Rhythmic'),
                                 ('1 segment (5-6s)', 'Relatively Rhythmic'),
                                 ('2 segments or 1 >6s', 'Less Rhythmic'),
                                 ('>2 segments or >10s', 'Non-Rhythmic')]:
            interpretation = rhythm_interpretations.get(category, '')
            rows.append(f"""
                <tr>
                    <td>{criteria}</td>
                    <td>{category}</td>
                    <td>{interpretation}</td>
                </tr>
            """)
        return '\n'.join(rows)
    
    html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vocal Feature Analysis Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', system-ui, sans-serif;
                    background-color: #ffffff;
                    color: #212529;
                    margin: 0;
                    padding: 0;
                    line-height: 1.6;
                }}
                
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid #dee2e6;
                }}
                
                .feature-section {{
                    background-color: #ffffff;
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
                    border: 1px solid #dee2e6;
                }}
                
                .feature-title{{
                    font-size: 18px;
                    font-weight: 600;
                    margin-bottom: 15px;
                    color: #1E3658;
                    border-bottom: 1px solid #dee2e6;
                    padding-bottom: 8px;
                }}
                
                .feature-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 15px;
                    margin-bottom: 15px;
                }}
                
                .feature-value {{
                    display: flex;
                    justify-content: space-between;
                }}

                .reference-table {{
                    width: 100%;
                    margin-top: 15px;
                    border-collapse: collapse;
                }}
                
                .reference-table th, .reference-table td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #dee2e6;
                }}
                
                .chart-container {{
                    background-color: #ffffff;
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
                    border: 1px solid #dee2e6;
                }}
                
                .plot-vertical {{
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                
                .plot {{
                    width: 100%;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="color:#1a1c1f;" >Multimodal Audio Classification</h1>
                    <h2 style="color:#1a1c1f;" >Clinical Interpretation Report</h2>
                    <p style="font-size: 20px; margin: 15px 0;">Analysis for sample: <strong style="font-size: 20px;">{sample_name}</strong></p>
                </div>
                <!-- Pause Analysis Section -->
                <div class="feature-section">
                    <div class="feature-title">Pause Analysis</div>
                    <div class="feature-grid">
                        <div class="feature-value">
                            <span>Pause Count:</span>
                            <span><strong>{pause_count}</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Category:</span>
                            <span><strong>{pause_category}</strong></span>
                        </div>
                    </div>
                    <table class="reference-table">
                        <tr>
                            <th>Pause Count</th>
                            <th>Category</th>
                            <th>Interpretation</th>
                        </tr>
                        {create_pause_table()}
                    </table>
                </div>
                
                <!-- Energy Analysis Section-->
                <div class="feature-section">
                    <div class="feature-title">Energy of Frequency Domain Analysis</div>
                    <div class="feature-grid">
                        <div class="feature-value">
                            <span>Measured Value:</span>
                            <span><strong>{energy_analysis['value']:.2f} dB</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Category:</span>
                            <span><strong>{energy_analysis['category'].split(': ')[1]}</strong></span>
                        </div>
                    </div>
                    <table class="reference-table">
                        <tr>
                            <th>Energy of Frequency Domain Range</th>
                            <th>Category</th>
                            <th>Interpretation</th>
                        </tr>
                        {create_ranges_table(energy_analysis,'energy')}
                    </table>
                </div>
                <!-- Rhythmic Structure Section-->
                <div class="feature-section">
                    <div class="feature-title">Rhythmic Structure Analysis</div>
                    <div class="feature-grid">
                        <div class="feature-value">
                            <span>Flat Segments:</span>
                            <span><strong>{len(flat_segments)}</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Total Duration:</span>
                            <span><strong>{total_flat_duration:.2f} sec</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Longest Segment:</span>
                            <span><strong>{longest_flat:.2f} sec</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Category:</span>
                            <span><strong>{rhythm_category}</strong></span>
                        </div>
                    </div>
                    <table class="reference-table">
                        <tr>
                            <th>Criteria</th>
                            <th>Category</th>
                            <th>Interpretation</th>
                        </tr>
                        {create_rhythm_table()}
                    </table>
                </div>
                <!-- Shimmer Analysis Section-->
                <div class="feature-section">
                    <div class="feature-title">Shimmer Standard Deviation Analysis</div>
                    <div class="feature-grid">
                        <div class="feature-value">
                            <span>Measured Value:</span>
                            <span><strong>{shimmer_analysis['value']:.2f}</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Category:</span>
                            <span><strong>{shimmer_analysis['category'].split(': ')[1]}</strong></span>
                        </div>
                    </div>
                    <table class="reference-table">
                        <tr>
                            <th>Shimmer Standard Deviation Range</th>
                            <th>Category</th>
                            <th>Interpretation</th>
                        </tr>
                        {create_ranges_table(shimmer_analysis,'shimmer')}
                    </table>
                </div>
                
                <!-- Fundamental Frequency Analysis-->
                <div class="feature-section">
                    <div class="feature-title">Fundamental Frequency Analysis</div>
                    <div class="feature-grid">
                        <div class="feature-value">
                            <span>Measured Value:</span>
                            <span><strong>{f0_analysis['value']:.2f} KHz</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Category:</span>
                            <span><strong>{f0_analysis['category'].split(': ')[1]}</strong></span>
                        </div>
                    </div>
                    <table class="reference-table">
                        <tr>
                            <th>Frequency Range</th>
                            <th>Category</th>
                            <th>Interpretation</th>
                        </tr>
                        {create_ranges_table(f0_analysis,'f0')}
                    </table>
                </div>

                <!-- Formant Frequency Analysis -->
                <div class="feature-section">
                    <div class="feature-title">Third Formant Frequency Analysis</div>
                    <div class="feature-grid">
                        <div class="feature-value">
                            <span>Measured Value:</span>
                            <span><strong>{f3_analysis['value']:.2f} KHz</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Category:</span>
                            <span><strong>{f3_analysis['category'].split(': ')[1]}</strong></span>
                        </div>
                    </div>
                    <table class="reference-table">
                        <tr>
                            <th>Frequency Range</th>
                            <th>Category</th>
                            <th>Interpretation</th>
                        </tr>
                        {create_ranges_table(f3_analysis,'f3')}
                    </table>
                </div>
            </div>
        </body>
        </html>
    """

    return features_index , html




def get_pause_interpret(quartile):
    groups = ['No pauses', 'Single pause', 'Few pauses', 'Several pauses']
    return f'{groups[quartile]} (e.g., pauses before nouns) were detected.'

def get_energy_interpret(quartile):
    groups = ['Very low', 'Low', 'Moderate', 'High']
    return f'{groups[quartile]} energy level were observed in the voice (see Spectrogram highlighted in green).'

def get_entropy_interpret(quartile):
    groups = ['rhythmic, with no', 'relatively rhythmic, with minimal', 'less rhythmic, with some evidence of', 'non-rhythmic, with strong evidence of']
    return f'Speech was {groups[quartile]} flat or monotonous segments (see Entropy curve)'

def get_shimmer_interpret(quartile):
    groups = [['stable', 'low'], ['almost stable', 'relatively low'], ['almost unstable', 'relatively high'], ['unstable', 'high']]
    return f'The voice was {groups[quartile][0]} (standard deviation of Shimmer was {groups[quartile][1]}).'

def get_f0_interpret(quartile):
    groups = ['poor, with very flat or monotonous speech', 'limited, with slightly reduced pitch variation',
              'relatively acceptable, with natural pitch modulation', 'acceptable, with strong pitch dynamics, energetic and engaging voice']
    return f'Control over vocal folds was {groups[quartile]} (see F0 curve).'

def get_f3_interpred(quartile):
    groups = ['poor', 'limited', 'natural', 'precise']
    return f'Tongue–lip coordination for phoneme and syllable production was {groups[quartile]} (see F3 curve).'

def get_final_decision(pred_label):
    if pred_label == 0:
        return 'Overall, there were minimal signs of cognitive impairment, suggesting the individual is likely cognitively healthy.'
    elif pred_label == 1:
        return 'Overall, there were some signs of cognitive impairment, suggesting the individual has likely mild cognitive impairment.'
    else:
        return 'Overall, there were several signs of cognitive impairment, suggesting the individual is likely cognitively impaired.'
    
def vocal_analysis_interpretation_report(sample_name: str, features: dict, pred_label: int) -> str:
    """
    Generate an HTML report interpreting vocal features with bullet points.
    
    Args:
        sample_name: Name of the sample being analyzed
        features: Dictionary containing quartile indices for each feature:
            {
                'pause': int (0-3),
                'energy': int (0-3),
                'entropy': int (0-3),
                'shimmer': int (0-3),
                'f0': int (0-3),
                'f3': int (0-3)
            }
        pred_label: Final prediction label (0-2)
    
    Returns:
        str: HTML report as a string
    """
    # Generate interpretations
    interpretations = {
        'pause': get_pause_interpret(features['pause']),
        'energy': get_energy_interpret(features['energy']),
        'entropy': get_entropy_interpret(features['entropy']),
        'shimmer': get_shimmer_interpret(features['shimmer']),
        'f0': get_f0_interpret(features['f0']),
        'f3': get_f3_interpred(features['f3'])
    }
    
    # Generate bullet points HTML
    bullet_points = "\n".join([
        f'<li class="interpretation-item">{interpretations[feature]}</li>'
        for feature in ['pause', 'energy', 'entropy', 'shimmer', 'f0', 'f3']
    ])
    
    html =  f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vocal Feature Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', system-ui, sans-serif;
                background-color: #ffffff;
                color: #212529;
                margin: 0;
                padding: 0;
                line-height: 1.6;
            }}
            
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #dee2e6;
            }}
            
            .feature-section {{
                background-color: #ffffff;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.05);
                border: 1px solid #dee2e6;
            }}
            
            .feature-title {{
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 15px;
                color: #1E3658;
                border-bottom: 1px solid #dee2e6;
                padding-bottom: 8px;
            }}
            
            .interpretation-list {{
                padding-left: 20px;
                margin: 0;
            }}
            
            .interpretation-item {{
                margin-bottom: 10px;
                position: relative;
                list-style-type: none;
                padding-left: 25px;
                font-size: 17px;
            }}
            
            .interpretation-item:before {{
                content: "•";
                color: #26A69A;
                font-size: 24px;
                position: absolute;
                left: 0;
                top: -2px;
            }}
            
            .final-decision {{
                background-color: #ffffff;
                border-radius: 12px;
                padding: 25px;
                margin-top: 30px;
                border-left: 4px solid #1E3658;
                font-size: 17px;
                line-height: 1.7;
                box-shadow: 0 4px 20px rgba(0,0,0,0.05);
                border: 1px solid #dee2e6;
            }}
            
            .decision-label {{
                font-weight: 600;
                color: #1E3658;
                margin-bottom: 10px;
                display: block;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 style="color:#1a1c1f;" >Vocal Feature Analysis</h1>
                <h2 style="color:#1a1c1f;">Clinical Interpretation Report</h2>
                <p style="font-size: 20px; margin: 15px 0; color:#1a1c1f;">Analysis for sample: <strong style="font-size: 20px;">{sample_name}</strong></p>
            </div>
            
            <!-- Vocal Feature Interpretations -->
            <div class="feature-section">
                <div class="feature-title">Vocal Feature Analysis Summary</div>
                <ul class="interpretation-list">
                    {bullet_points}
                </ul>
            </div>
            
            <!-- Final Decision -->
            <div class="final-decision">
                <span class="decision-label">Clinical Interpretation:</span>
                {get_final_decision(pred_label)}
            </div>
        </div>
    </body>
    </html>
    """
    return interpretations , html