import matplotlib.pyplot as plt
import base64
from io import BytesIO
import matplotlib.patheffects as patheffects


def generate_pie_chart(contributions):
    """
    Generate a pie chart of modality contributions.
    
    Args:
        contributions: list - Modality contribution weights
        
    Returns:
        str - Base64 encoded PNG image of pie chart
    """
    # Store original matplotlib style to restore later
    original_style = plt.style.available[0]
    
    try:
        # Create the pie chart for modality contributions
        with plt.style.context('seaborn-v0_8-bright'):
            fig = plt.figure(figsize=(6, 6), facecolor='#0d1117')
            ax = fig.add_subplot(111)

            pie_colors = ['#008080','#457b9d', '#e76f51']  # Teal, Blue, Orange
            wedges, texts, autotexts = ax.pie(
                contributions,
                labels=['Acoustic', 'Linguistic', 'Demographic'],
                colors=pie_colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 12, 'color': 'white'},
                wedgeprops={'edgecolor': '#0d1117', 'linewidth': 1.5},
                explode=(0.05, 0.05, 0.05)
            )

            ax.set_title('Modality Contributions', fontsize=14, pad=20, color='white', fontweight='bold')
            plt.setp(autotexts, size=12, weight="bold", color='white',
                    path_effects=[patheffects.withStroke(linewidth=2, foreground='#333333')])

            # Save plot to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=120)
            buffer.seek(0)
            pie_chart_data = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()

        return pie_chart_data

    finally:
        # Restore original matplotlib style
        plt.style.use(original_style)