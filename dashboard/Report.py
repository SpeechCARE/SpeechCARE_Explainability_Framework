import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import os
from matplotlib import patheffects
from matplotlib.patheffects import withStroke

def generate_prediction_report(model, audio_path, demography_info, config):
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
            
            # Create acoustic analysis images (placeholder - replace with your actual plots)
            acoustic_images = []
            for i in range(2):
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#0d1117')
                ax.set_facecolor('#161b22')
                
                # Example plot (replace with your actual acoustic analysis)
                if i == 0:
                    # Spectrogram-like plot
                    data = np.random.rand(10, 50)
                    im = ax.imshow(data, cmap='viridis', aspect='auto')
                    plt.colorbar(im, ax=ax)
                    ax.set_title('Acoustic Feature Analysis', color='white', pad=10)
                else:
                    # Waveform-like plot
                    x = np.linspace(0, 10, 100)
                    y = np.sin(x) * np.exp(-x/10)
                    ax.plot(x, y, color='#FFA726')
                    ax.set_title('Pitch Contour Analysis', color='white', pad=10)
                
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#30363d')
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=100)
                buffer.seek(0)
                acoustic_images.append(base64.b64encode(buffer.read()).decode('utf-8'))
                plt.close()
    
    finally:
        # Restore original matplotlib style
        plt.style.use(original_style)
    
    html = f"""<!DOCTYPE html>
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
            
            /* ... (keep all your existing styles) ... */
            
            /* Enhanced acoustic section styles */
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
                margin-top: 15px;
            }}
            
            .image-container {{
                background-color: var(--card-bg);
                padding: 15px;
                border-radius: 8px;
                border: 1px solid var(--border-color);
                text-align: center;
            }}
            
            .image-container img {{
                max-width: 100%;
                border-radius: 6px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            }}
            
            .image-caption {{
                margin-top: 10px;
                font-size: 14px;
                color: var(--text-color);
                opacity: 0.8;
            }}
            
        </style>
    </head>
    <body>
        <div class="container">
            <!-- ... (keep all your existing HTML structure) ... -->

            <!-- Enhanced Acoustic Explainability Section -->
            <div class="explainability-section" id="acoustic-section">
                <div class="section-header" onclick="toggleSection('acoustic-section')">
                    <h3 class="section-title">Acoustic Explainability Module</h3>
                    <div class="toggle-icon"></div>
                </div>
                <div class="section-content">
                    <div class="acoustic-description">
                        <p>The acoustic analysis reveals important patterns in speech characteristics that contribute to the model's decision. 
                        These visualizations highlight key acoustic features such as pitch variation, speaking rate, and spectral properties 
                        that differ between cognitive health groups.</p>
                    </div>
                    
                    <div class="image-grid">
                        <div class="image-container">
                            <img src="data:image/png;base64,{acoustic_images[0]}" alt="Acoustic Feature Analysis">
                            <div class="image-caption">Figure 1: Spectrogram analysis showing frequency distribution over time</div>
                        </div>
                        
                        <div class="image-container">
                            <img src="data:image/png;base64,{acoustic_images[1]}" alt="Pitch Contour Analysis">
                            <div class="image-caption">Figure 2: Fundamental frequency (pitch) contour with smoothing</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ... (rest of your HTML) ... -->
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
    </html> """
    
    return html