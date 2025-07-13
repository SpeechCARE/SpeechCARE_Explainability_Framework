from explainability.plotting.explainability_plotting import plot_spectrogram
import base64
import os
import re



def generate_html_tutorial_page(
    examples,
    page_title="Spectrogram Interpretation Tutorial",
    header_text="This tutorial introduces clinicians to interpreting audio spectrograms through a series of labeled examples.",
):

    html_parts = [f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>{page_title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 2em;
            margin-bottom: 4em;
            background: #f4f4f9;
            color: #333;
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
        }}
        .tutorial-header {{
            max-width: 800px;
            margin: 0 auto 2em auto;
            font-size: 1.1em;
            text-align: center;
        }}
        details {{
            margin-bottom: 2em;
            background: #ffffff;
            border: 1px solid #ccc;
            border-radius: 6px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
            overflow: hidden;
        }}
        summary {{
            font-size: 1.1em;
            font-weight: bold;
            padding: 1em;
            cursor: pointer;
            user-select: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #1E3658;
            border-radius: 6px 6px 0 0;
        }}
        summary::-webkit-details-marker {{
            display: none;
        }}
        .summary-title {{
            flex: 1;
            text-align: left;
            color: white;
        }}
        .summary-icon {{
            font-size: 1.2em;
            margin-left: 10px;
            transition: transform 0.2s ease-in-out;
            color: white;
        }}
        details[open] .summary-icon {{
            transform: rotate(-90deg);
        }}
        .content {{
            padding: 1em 1em;
            margin-left: auto;
            margin-right: auto;
            max-width: 1000px;
            width: 100%;
            text-align: left;
        }}
        .section-block {{
            margin-bottom: 1.5em;
        }}
        .section-block label {{
            font-weight: bold;
            display: block;
            margin-bottom: 0.3em;
            color: #1E3658; }}/* Dark blue for section labels */

        audio{{
            width: 100%;
            height: 30px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ccc;
        }}
        .explanation-box {{
            background: #fefefe;
            border: 1px solid #2c3e50;
            padding: 1em;
            border-radius: 5px;
            font-size: 1em;
            line-height: 1.6em;
            max-width: 1000px;
        }}
        /* Dark blue bullets */
        .explanation-box ul{{
            padding-left: 1.2em;
            margin: 0.5em 0;
        }}
        .explanation-box li {{
            margin-bottom: 0.5em;
        }}
        .explanation-box li::marker {{
            color: #1E3658; /* Dark blue bullet color */
        }}
        /* Fallback for older browsers */
        .explanation-box ul li::before {{
            content: "•";
            color: #1E3658;
            display: inline-block;
            width: 1em;
            margin-left: -1em;
        }}
    </style>
    </head>
    <body>
    <h1>{page_title}</h1>
    <div class="tutorial-header">{header_text}</div>
        """]

    for idx, ex in enumerate(examples):
        html_parts.append(f"""
            <details>
                <summary>
                    <span class="summary-title">{ex['title']}</span>
                    <span class="summary-icon">▼</span>
                </summary>
                <div class="content">
                    <div class="section-block">
                    <label>Audio Sample:</label>
                    <audio controls>
                        <source src="data:audio/mp3;base64,{ex['audio_b64']}" type="audio/mp3">
                        Your browser does not support the audio element.
                    </audio>
                    </div>

                    <div class="section-block">
                    <label>Spectrogram:</label>
                    <img src="data:image/png;base64,{ex['spectrogram_b64']}" alt="Spectrogram for {ex['title']}">
                    </div>

                    <div class="section-block">
                    <label>Explanation:</label>
                    <div class="explanation-box">{ex['explanation']}</div>
                    </div>
                </div>
            </details>
        """)

    html_parts.append("</body>\n</html>")
    return ''.join(html_parts)


def encode_audio_to_base64(audio_path):
    """
    Reads an audio file and encodes it into base64.
    """
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def format_interpretation_to_html(text):
    """
    Converts markdown-style interpretation text into an HTML unordered list.
    """
    lines = text.strip().split("\n")
    list_items = []

    for line in lines:
        line = line.strip()
        if line.startswith("-"):
            line = line[1:].strip()
            # Replace **bold** with <strong> tags
            line = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", line)
            list_items.append(f"<li>{line}</li>")

    return "<ul>\n" + "\n".join(list_items) + "\n</ul>"

def get_title_and_key(audio_filename):
    """
    Returns the interpretation key and human-readable title for a given filename.
    """
    if "rain" in audio_filename:
        return "rain", "Raindrops Falling on a Surface"
    elif "office" in audio_filename:
        return "office", "Chalk Writing on a Blackboard"
    elif "pet" in audio_filename:
        return "pet", "Dog Barking Repeatedly"
    elif "speak" in audio_filename:
        return "speak", "Woman Counting"
    elif "step" in audio_filename:
        return "step", "Footsteps on a Hard Floor"
    elif "traffic" in audio_filename:
        return "traffic", "Bell Ringing in a Street Scene"
    elif "dish" in audio_filename:
        return "dish", "Dishes Collision"
    else:
        return None, None

def build_example_list(audio_dir, interpretation):
    """
    Builds a list of dictionaries for each audio file with relevant data.
    """
    example_list = []

    for filename in os.listdir(audio_dir):
        filepath = os.path.join(audio_dir, filename)
        key, title = get_title_and_key(filename)

        if key is None:
            continue  # Skip files without a valid key

        audio_b64 = encode_audio_to_base64(filepath)
        spectrogram_b64 = plot_spectrogram(audio_path=filepath,use_mel=False,add_colorbar=True,return_base64=True)
        explanation_html = format_interpretation_to_html(interpretation[key])

        example_list.append({
            "title": title,
            "audio_b64": audio_b64,
            "spectrogram_b64": spectrogram_b64,
            "explanation": explanation_html
        })

    return example_list
