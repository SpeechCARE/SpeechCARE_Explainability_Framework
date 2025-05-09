import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import os
from matplotlib import patheffects
from matplotlib.patheffects import withStroke

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

            .image-grid {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 20px;
            }}

            .image-container {{
                background-color: var(--card-bg);
                padding: 15px;
                border-radius: 8px;
                border: 1px solid var(--border-color);
                text-align: center;
            }}

            .image-caption {{
                margin-top: 10px;
                font-size: 14px;
                color: var(--text-color);
                opacity: 0.8;
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
                            <img src={acoustic_data['spectrogram_image']} 
                                alt="Acoustic Feature Analysis" 
                                style="width: 100%; border-radius: 6px;">
                            <div class="image-caption">Figure 1: Spectrogram analysis</div>
                        </div>
                        
                        <!-- Second image -->
                        <div class="image-container">
                            <img src={acoustic_data['entropy_image']} 
                                alt="Entropy Analysis" 
                                style="width: 100%; border-radius: 6px;">
                            <div class="image-caption">Figure 2: Entropy analysis</div>
                        </div>
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

 