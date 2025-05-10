import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import os
from matplotlib import patheffects
from matplotlib.patheffects import withStroke
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
from typing import List, Tuple,Dict
import librosa

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
        return "Rhythmic"
    
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

def generate_spectrogram_plot_for_html(
        modified_mel_spectrogram: np.ndarray,
        pauses: List[Tuple[float, float, str, float, float, float, bool]],
        formant_values: dict,
        sr: int = 16000,
        figsize: Tuple[int, int] = (20, 4),
        dpi: int = 120
    ) -> str:
    """
    Generate a base64-encoded spectrogram plot with pauses and formants for HTML embedding.
    
    Args:
        modified_mel_spectrogram: The modified mel spectrogram (in dB)
        pauses: List of pause tuples (start, end, pause_type, duration, mean_pitch, mean_intensity, mark)
        formant_values: Dictionary of formant values {'F0': [], 'F1': [], etc.}
        sr: Sampling rate
        figsize: Figure dimensions (width, height)
        dpi: Figure resolution
        
    Returns:
        Base64-encoded PNG image string
    """
    # Create figure with dark background
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('#0d1117')  # Dark background
    ax.set_facecolor('#161b22')  # Dark card background
    
    # Plot spectrogram
    img = librosa.display.specshow(
        modified_mel_spectrogram, 
        sr=sr, 
        x_axis="time", 
        y_axis="mel", 
        cmap="viridis", 
        ax=ax
    )
    
    # Style the plot
    ax.tick_params(axis='both', colors='white')
    ax.spines['bottom'].set_color('#404040')
    ax.spines['left'].set_color('#404040')
    ax.set_xlabel("Time (ms)", color='white', fontsize=12)
    ax.set_ylabel("Frequency (Hz)", color='white', fontsize=12)
    
    # Set x-axis ticks in milliseconds
    audio_duration = modified_mel_spectrogram.shape[1] * (512 / sr)  # hop_length=512
    time_ticks_ms = np.arange(0, audio_duration * 1000, 500)  # Every 500 ms
    time_ticks_seconds = time_ticks_ms / 1000
    ax.set_xticks(time_ticks_seconds)
    ax.set_xticklabels([f"{int(t)}" for t in time_ticks_ms], color='white', rotation=45)
    
    # Get max mel frequency for pause plotting
    max_mel = img.axes.yaxis.get_data_interval()[-1]
    
    # Plot pauses with different styles
    if pauses:
        # Create proxy artists for legend
        ax.plot([], [], color="yellow", linestyle="-", linewidth=2, label="Informative Pause")
        ax.plot([], [], color="yellow", linestyle="--", linewidth=2, label="Natural Pause")
        
        for start, end, _, _, _, _, mark in pauses:
            linestyle = "-" if mark else "--"
            ax.plot(
                [start, start, end, end, start],
                [0, max_mel, max_mel, 0, 0],
                color="yellow",
                linestyle=linestyle,
                linewidth=2
            )
    
    # Plot formants if available
    formant_colors = {"F0": 'red', "F1": 'cyan', "F2": 'white', "F3": '#FF8C00'}
    times = np.linspace(0, audio_duration, len(formant_values.get('F1', [])))
    
    for formant, values in formant_values.items():
        if formant == "F0":
            # F0 has different time stamps
            time_stamps = np.linspace(0, audio_duration, len(values))
            ax.plot(
                time_stamps,
                values,
                label=formant,
                linewidth=3,
                color=formant_colors[formant]
            )
        elif formant in ["F3"]:
            ax.plot(
                times,
                values,
                label=formant,
                linewidth=2,
                color=formant_colors[formant]
            )
    
    if formant_values:
        legend = ax.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d')
        for text in legend.get_texts():
            text.set_color('white')
    
    plt.tight_layout()
    
    # Save to buffer
    buffer = BytesIO()
    plt.savefig(
        buffer, 
        format='png', 
        facecolor=fig.get_facecolor(), 
        bbox_inches='tight', 
        dpi=dpi
    )
    buffer.seek(0)
    plt.close(fig)
    
    return base64.b64encode(buffer.read()).decode('utf-8')
def generate_entropy_plot_for_html(
        times: np.ndarray,
        smoothed_entropy: np.ndarray,
        flat_segments: List[Tuple[float, float]],
        figsize: Tuple[int, int] = (20, 2),
        dpi: int = 120
    ) -> str:
    """
    Generate a base64-encoded entropy plot for HTML embedding.
    
    Args:
        times: Array of time points (in seconds)
        smoothed_entropy: Array of entropy values
        flat_segments: List of (start, end) tuples for flat segments
        figsize: Figure dimensions (width, height)
        dpi: Figure resolution
        
    Returns:
        Base64-encoded PNG image string
    """
    # Create figure with dark background
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('#0d1117')  # Dark background
    
    ax = plt.gca()
    ax.set_facecolor('#161b22')  # Dark card background
    
    # Plot entropy curve
    plt.plot(times[:len(smoothed_entropy)], smoothed_entropy, 
             label='Smoothed Shannon Entropy', color='#1E88E5', linewidth=1.5)
    
    # Highlight flat segments
    for start, end in flat_segments:
        plt.axvspan(start, end, color='#F44336', alpha=0.3, 
                   label='Flat Segment' if start == flat_segments[0][0] else "")
    
    # Customize appearance
    plt.xticks(np.arange(0, np.max(times), step=1), color='white')
    plt.yticks(color='white')
    plt.grid(axis='x', color='#30363d', linestyle='--', alpha=0.5)
    plt.xlabel('Time (seconds)', color='white')
    plt.ylabel('Entropy (bits)', color='white')
    plt.title('Spectral Entropy Analysis', color='white', pad=20, fontweight='bold')
    
    # Style the spines
    for spine in ax.spines.values():
        spine.set_color('#404040')
    
    # Add legend with white text
    legend = plt.legend(facecolor='#161b22', edgecolor='#30363d')
    for text in legend.get_texts():
        text.set_color('white')
    
    plt.tight_layout()
    
    # Save to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', facecolor=fig.get_facecolor(), 
               bbox_inches='tight', dpi=dpi)
    buffer.seek(0)
    plt.close(fig)
    
    return base64.b64encode(buffer.read()).decode('utf-8')
def generate_prediction_report(model, audio_path, demography_info,acoustic_data,linguistic_data, config):
    """Generate an interactive HTML report with dark/light mode toggle.
    
    Args:
        model: Loaded TBNet model
        audio_path: Path to input audio file
        demography_info: Demographic data (e.g., age)
        config: Model configuration parameters
        
    Returns:
        HTML string containing the interactive report
    """
    
    # Store original matplotlib style to restore later
    original_style = plt.style.available[0]  # Default style
    plot_data_entropy = generate_entropy_plot_for_html(acoustic_data['entropy']['times'],
                                                        acoustic_data['entropy']['smoothed_entropy'],
                                                        acoustic_data['entropy']['flat_segments'])
    
    plot_data_spectrogram = generate_spectrogram_plot_for_html(
                                                    modified_mel_spectrogram=acoustic_data['spectrogram']['modified_log_S'],
                                                    pauses=acoustic_data['spectrogram']['pauses'],
                                                    formant_values=acoustic_data['spectrogram']['formant_values'],
                                                    sr=acoustic_data['spectrogram']['sr']
                                                )

    total_flat_duration = sum(end - start for start, end in acoustic_data['entropy']['flat_segments'])
    longest_flat = max([end - start for start, end in acoustic_data['entropy']['flat_segments']] or [0])
    rhythm_index , rhythm_category = categorize_rhythmic_structure(acoustic_data['entropy']['flat_segments'])
    pause_index , pause_category = categorize_pauses(len(acoustic_data['spectrogram']['pauses']))
    shimmer_index = int(acoustic_data['shimmer_analysis']['category'].split(':')[0][1:]) - 1
    energy_index = int(acoustic_data['shimmer_analysis']['category'].split(':')[0][1:]) - 1
    f0_index = int(acoustic_data['f0_analysis']['category'].split(':')[0][1:]) - 1
    f3_index = int(acoustic_data['f3_analysis']['category'].split(':')[0][1:]) - 1
    features_index = {
    'pause': pause_index,
    'energy': energy_index,
    'entropy': rhythm_index,
    'shimmer': shimmer_index,
    'f0': f0_index,
    'f3': f3_index
    }
    
    try:
        # Run inference and get the gating weights
        predicted_label, probabilities = model.inference(audio_path, demography_info, config)
        
        # Get modality weights (ensure your model stores these)
        if hasattr(model, 'last_gate_weights'):
            gate_weights = model.last_gate_weights[0].tolist()
        else:
            gate_weights = [0.4, 0.4, 0.2]  # fallback
        
        # Prepare data
        class_names = ['Control', 'MCI', 'AD']
        prob_values = [prob * 100 for prob in probabilities]
        modalities = ['Acoustic', 'Linguistic', 'Demographic']
        predicted_class = class_names[predicted_label]
        
        # Create plots with dark style only for this figure
        with plt.style.context('dark_background'):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1]})
            fig.patch.set_facecolor('#0d1117')  # Dark background
            
            # Prediction bar chart (left)
            bar_colors = ['#4CAF50','#ede28c', '#9c4940']    # accent green , orange , accent red
            bars = ax1.bar(class_names, prob_values, color=bar_colors, 
                          edgecolor='white', linewidth=0.5, alpha=1)
            ax1.set_title('Prediction Confidence', fontsize=14, pad=20, color='white', fontweight='bold')
            ax1.set_ylabel('Probability (%)', fontsize=12, color='#b0b0b0')
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='both', colors='white')
            ax1.tick_params(axis='x', which='both', labelsize=13, colors='white')
            ax1.spines['bottom'].set_color('#404040')
            ax1.spines['left'].set_color('#404040')
            
            # Add value labels with glow effect
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom',
                        color='white', fontsize=11, fontweight='bold',
                        path_effects=[patheffects.withStroke(linewidth=3, foreground='#333333')])
            # Remove vertical grid lines
            ax1.grid(axis='x', visible=False)

            # Modality pie chart (right) - Orange accent theme
            pie_colors = ['#008080','#457b9d', '#e76f51']  # Teal, Blue,Orange
            wedges, texts, autotexts = ax2.pie(
                gate_weights,
                labels=modalities,
                colors=pie_colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 12, 'color': 'white'},
                wedgeprops={'edgecolor': '#0d1117', 'linewidth': 1.5},
                explode=(0.05, 0.05, 0.05)  # Slight separation
            )
            ax2.set_title('Modality Contributions', fontsize=14, pad=20, color='white', fontweight='bold')
            
            # Make percentages bold and larger
            plt.setp(autotexts, size=12, weight="bold", color='white',
                    path_effects=[patheffects.withStroke(linewidth=2, foreground='#333333')])
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.3)
            
            # Save plot
            buffer = BytesIO()
            plt.savefig(buffer, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=120)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
    
    finally:
        # Restore original matplotlib style
        plt.style.use(original_style)
    
    # Generate HTML with dark/light mode toggle
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Decision Analysis</title>
        <style>
            :root {{
                --bg-color: #0d1117;
                --text-color: #e6edf3;
                --card-bg: #161b22;
                --border-color: #30363d;
                --highlight: #FFA726;
                --accent-blue: #1E88E5;
                --accent-green: #4CAF50;
                --accent-teal: #26A69A;
                --accent-red: #F44336;
            }}
            
            [data-theme="light"] {{
                --bg-color: #f8f9fa;
                --text-color: #212529;
                --card-bg: #ffffff;
                --border-color: #dee2e6;
                --highlight: #FF7043;
            }}
            
            body {{
                font-family: 'Segoe UI', system-ui, sans-serif;
                background-color: var(--bg-color);
                color: var(--text-color);
                margin: 0;
                padding: 0;
                transition: all 0.3s ease;
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
                border-bottom: 1px solid var(--border-color);
                position: relative;
            }}
            
            .theme-toggle {{
                position: absolute;
                right: 0;
                top: 0;
                background: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 20px;
                padding: 5px 10px;
                cursor: pointer;
                font-size: 14px;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            
            .result-card {{
                background-color: var(--card-bg);
                border-radius: 12px;
                padding: 25px;
                margin-bottom: 30px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                border: 1px solid var(--border-color);
                text-align: center;
            }}
            
            .prediction {{
                font-size: 24px;
                font-weight: 600;
                margin-bottom: 10px;
                color: var(--highlight);
            }}
            
            .confidence {{
                font-size: 18px;
                color: var(--text-color);
                opacity: 0.9;
            }}
            
            .chart-container {{
                background-color: var(--card-bg);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                border: 1px solid var(--border-color);
            }}
            
            .details-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 25px;
                margin-bottom: 30px;
            }}
            
            .detail-card {{
                background-color: var(--card-bg);
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                border: 1px solid var(--border-color);
            }}
            
            .detail-title {{
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 15px;
                color: var(--highlight);
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 8px;
            }}
            
            .modality-item {{
                display: flex;
                align-items: center;
                margin: 12px 0;
                padding: 10px;
                border-radius: 8px;
                background-color: rgba(255,255,255,0.05);
            }}
            
            .modality-color {{
                width: 24px;
                height: 24px;
                border-radius: 6px;
                margin-right: 12px;
                flex-shrink: 0;
            }}
            
            .modality-value {{
                font-weight: 600;
                margin-left: auto;
            }}
            
            .audio-info {{
                background-color: var(--card-bg);
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                border: 1px solid var(--border-color);
            }}
            
            .info-title {{
                font-weight: 600;
                margin-bottom: 8px;
                color: var(--accent-blue);
            }}
            .acoustic-description {{
                margin-bottom: 20px;
                padding: 15px;
                background-color: rgba(255,255,255,0.03);
                border-radius: 8px;
                border-left: 4px solid var(--accent-teal);
            }}
            .explainability-section {{
                background-color: var(--card-bg);
                border-radius: 12px;
                padding: 0;
                margin-bottom: 20px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                border: 1px solid var(--border-color);
                overflow: hidden;
            }}
            .feature-section {{
                background-color: var(--card-bg);
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                border: 1px solid var(--border-color);
            }}
            
            .feature-title {{
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 15px;
                color: var(--accent-blue);
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 8px;
            }}
            
            .feature-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 15px;
            }}
            
            .feature-value {{
                display: flex;
                justify-content: space-between;
                padding: 8px 12px;
                background-color: rgba(255,255,255,0.03);
                border-radius: 6px;
                border-left: 3px solid var(--accent-teal);
            }}
            
            .reference-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
                font-size: 14px;
            }}
            
            .reference-table th {{
                background-color: rgba(255,255,255,0.05);
                padding: 12px 15px;
                text-align: left;
                font-weight: 600;
                color: var(--accent-teal);
                border-bottom: 2px solid var(--border-color);
            }}
            
            .reference-table td {{
                padding: 10px 15px;
                border-bottom: 1px solid var(--border-color);
            }}
            
            .reference-table tr:last-child td {{
                border-bottom: none;
            }}
            
            .reference-table tr:hover {{
                background-color: rgba(255,255,255,0.03);
            }}
            
            
            
            .section-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px;
                cursor: pointer;
                user-select: none;
            }}
            
            .section-title {{
                font-size: 18px;
                font-weight: 600;
                color: var(--accent-teal);
                margin: 0;
            }}
            
            .toggle-icon {{
                width: 20px;
                height: 20px;
                position: relative;
                transition: transform 0.3s ease;
            }}
            
            .toggle-icon::before,
            .toggle-icon::after {{
                content: '';
                position: absolute;
                background-color: var(--text-color);
                transition: all 0.3s ease;
            }}
            
            .toggle-icon::before {{
                top: 50%;
                left: 0;
                right: 0;
                height: 2px;
                transform: translateY(-50%);
            }}
            
            .toggle-icon::after {{
                top: 0;
                left: 50%;
                bottom: 0;
                width: 2px;
                transform: translateX(-50%);
            }}
            
            .collapsed .toggle-icon::after {{
                transform: translateX(-50%) rotate(90deg);
                opacity: 1;
            }}
            
            .section-content {{
                padding: 0 20px;
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease, padding 0.3s ease;
                border-top: 1px solid transparent;
            }}
            
            .expanded .section-content {{
                padding: 0 20px 20px;
                max-height: 2000px;
                border-top: 1px solid var(--border-color);
            }}
        
          
            .transcription {{
                background-color: rgba(255,255,255,0.05);
                padding: 15px;
                border-radius: 8px;
                font-style: italic;
                line-height: 1.5;
            }}
         
            @media (max-width: 768px) {{
                .details-grid {{
                    grid-template-columns: 1fr;
                }}

                .feature-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .reference-table {{
                    display: block;
                    overflow-x: auto;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Model Decision Analysis</h1>
                <p style="text-align: center;" >Comprehensive breakdown for: {os.path.basename(audio_path)}</p>
                <div class="theme-toggle" onclick="toggleTheme()">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                    </svg>
                    <span>Dark Mode</span>
                </div>
            </div>
            
            <div class="result-card">
                <div class="prediction">Predicted: <span style="color: {bar_colors[predicted_label]}">{predicted_class}</span></div>
                <div class="confidence">Confidence: {prob_values[predicted_label]:.1f}%</div>
            </div>
            
          
            
           <div class="details-grid">
                <div class="detail-card">
                    <div class="detail-title">Prediction Confidence</div>
                    <div class="modality-item" style="border-left: 4px solid var(--accent-green);">
                        <div>Control</div>
                        <div class="modality-value">{prob_values[0]:.1f}%</div>
                    </div>
                    <div class="modality-item" style="border-left: 4px solid #ede28c;">
                        <div>Mild Cognitive Impairment</div>
                        <div class="modality-value">{prob_values[1]:.1f}%</div>
                    </div>
                    <div class="modality-item" style="border-left: 4px solid #9c4940;">
                        <div>Alzheimer's Disease</div>
                        <div class="modality-value">{prob_values[2]:.1f}%</div>
                    </div>
                </div>
                
                
                <div class="detail-card">
                    <div class="detail-title">Modality Contributions</div>
                    <div class="modality-item">
                        <div class="modality-color" style="background-color: #008080;"></div>
                        <div>Acoustic Analysis</div>
                        <div class="modality-value">{gate_weights[0]*100:.1f}%</div>
                    </div>
                    <div class="modality-item">
                        <div class="modality-color" style="background-color: #457b9d;"></div>
                        <div>Linguistic Features</div>
                        <div class="modality-value">{gate_weights[1]*100:.1f}%</div>
                    </div>
                    <div class="modality-item">
                        <div class="modality-color" style="background-color: #e76f51;"></div>
                        <div>Demographic Factors</div>
                        <div class="modality-value">{gate_weights[2]*100:.1f}%</div>
                    </div>
                </div>
            </div>

            <div class="chart-container">
                <img src="data:image/png;base64,{plot_data}" alt="Analysis Results" style="width: 100%; border-radius: 8px;">
            </div>

            <!-- Acoustic Explainability Section -->
            <div class="explainability-section" id="acoustic-section">
                <div class="section-header" onclick="toggleSection('acoustic-section')">
                    <h3 class="section-title">Acoustic Explainability Module</h3>
                    <div class="toggle-icon"></div>
                </div>
                <div class="section-content">
                    <div class="acoustic-description">
                        <p>The acoustic analysis reveals patterns in speech characteristics that contribute to the model's decision. 
                        These visualizations highlight key acoustic features that differ between cognitive health groups.</p>
                    </div>
                    
                    <div class="image-grid">
                        <div class="image-container">
                            <img src="data:image/png;base64,{plot_data_spectrogram}" 
                            alt="Spectrogram Analysis" 
                            style="width: 100%; border-radius: 8px;">
                            <div class="image-caption">Figure 1: Spectrogram analysis</div>
                        </div>
                        
                        <!-- Second image -->
                        <div class="image-container">
                            <img src="data:image/png;base64,{plot_data_entropy}" alt="Entropy Analysis" style="width: 100%; border-radius: 8px;">

                            <div class="image-caption">Figure 2: Entropy analysis</div>
                        </div>
                    </div>

                    <!-- Pause Analysis Section -->
                    <div class="feature-section">
                        <div class="feature-title">Pause Analysis</div>
                        <div class="feature-grid">
                            <div class="feature-value">
                                <span>Pause Count:</span>
                                <span><strong>{len(acoustic_data['spectrogram']['pauses'])}</strong></span>
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
                                <span><strong>{acoustic_data['energy_analysis']['value']:.2f} dB</strong></span>
                            </div>
                            <div class="feature-value">
                                <span>Category:</span>
                                <span><strong>{acoustic_data['energy_analysis']['category'].split(': ')[1]}</strong></span>
                            </div>
                        </div>
                        <table class="reference-table">
                            <tr>
                                <th>Energy of Frequency Domain Range</th>
                                <th>Category</th>
                                <th>Interpretation</th>
                            </tr>
                            {create_ranges_table(acoustic_data['energy_analysis'],'energy')}
                        </table>
                    </div>
                    <!-- Rhythmic Structure Section-->
                    <div class="feature-section">
                        <div class="feature-title">Rhythmic Structure Analysis</div>
                        <div class="feature-grid">
                            <div class="feature-value">
                                <span>Flat Segments:</span>
                                <span><strong>{len(acoustic_data['entropy']['flat_segments'])}</strong></span>
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
                                <span><strong>{acoustic_data['shimmer_analysis']['value']:.2f}</strong></span>
                            </div>
                            <div class="feature-value">
                                <span>Category:</span>
                                <span><strong>{acoustic_data['shimmer_analysis']['category'].split(': ')[1]}</strong></span>
                            </div>
                        </div>
                        <table class="reference-table">
                            <tr>
                                <th>Shimmer Standard Deviation Range</th>
                                <th>Category</th>
                                <th>Interpretation</th>
                            </tr>
                            {create_ranges_table(acoustic_data['shimmer_analysis'],'shimmer')}
                        </table>
                    </div>
                
                    <!-- Fundamental Frequency Analysis-->
                    <div class="feature-section">
                        <div class="feature-title">Fundamental Frequency Analysis</div>
                        <div class="feature-grid">
                            <div class="feature-value">
                                <span>Measured Value:</span>
                                <span><strong>{acoustic_data['f0_analysis']['value']:.2f} KHz</strong></span>
                            </div>
                            <div class="feature-value">
                                <span>Category:</span>
                                <span><strong>{acoustic_data['f0_analysis']['category'].split(': ')[1]}</strong></span>
                            </div>
                        </div>
                        <table class="reference-table">
                            <tr>
                                <th>Frequency Range</th>
                                <th>Category</th>
                                <th>Interpretation</th>
                            </tr>
                            {create_ranges_table(acoustic_data['f0_analysis'],'f0')}
                        </table>
                    </div>

                    <!-- Formant Frequency Analysis -->
                    <div class="feature-section">
                        <div class="feature-title">Third Formant Frequency Analysis</div>
                        <div class="feature-grid">
                            <div class="feature-value">
                                <span>Measured Value:</span>
                                <span><strong>{acoustic_data['f3_analysis']['value']:.2f} KHz</strong></span>
                            </div>
                            <div class="feature-value">
                                <span>Category:</span>
                                <span><strong>{acoustic_data['f3_analysis']['category'].split(': ')[1]}</strong></span>
                            </div>
                        </div>
                        <table class="reference-table">
                            <tr>
                                <th>Frequency Range</th>
                                <th>Category</th>
                                <th>Interpretation</th>
                            </tr>
                            {create_ranges_table(acoustic_data['f3_analysis'],'f3')}
                        </table>
                    </div>
                </div>
            </div>

            <!-- New Linguistic Explainability Section -->
            <div class="explainability-section" id="linguistic-section">
                <div class="section-header" onclick="toggleSection('linguistic-section')">
                    <h3 class="section-title">Linguistic Explainability Module</h3>
                    <div class="toggle-icon"></div>
                </div>
                <div class="section-content">
                    <!-- Content will go here -->
                    <p>Linguistic analysis details will appear here...</p>
                </div>
            </div>
            
            <div class="audio-info">
                <div class="detail-title">Age</div>
                <p><strong>Age category:</strong> {demography_info}</p>
                
                <div class="detail-title" style="margin-top: 20px;">Transcript of Audio File</div>
                <div class="transcription">{model.transcription or "No transcription available"}</div>
            </div>
        </div>
        
        <script>
          
            function toggleSection(sectionId) {{
                const section = document.getElementById(sectionId);
                section.classList.toggle('expanded');
                section.classList.toggle('collapsed');
            }}

            // Initialize sections as collapsed
            document.addEventListener('DOMContentLoaded', function() {{
                document.getElementById('acoustic-section').classList.add('collapsed');
                document.getElementById('linguistic-section').classList.add('collapsed');
            }});
        </script>
    </body>
    </html>
    """
    
    return html

 