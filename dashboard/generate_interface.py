
import pandas as pd
from openai import OpenAI
import re
import base64

css_style = f"""
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}

                body {{
                    background-color: #E5F1F3;
                    font-family: Arial, Helvetica, sans-serif;
                    font-size: 1em;
                    line-height: 1.5;
                    color: #1E3658;
                }}
                
                /* Layout Components */
                .container{{
                    width: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                }}

                .vertical_box {{
                    width: 100%;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }}

                .small_box{{
                    width:80%;
                }}

                .horizontal_box {{
                    width: 100%;
                    display: flex;
                    flex-direction: row;
                    justify-content: space-between;
                    padding: 1rem;
                    gap: 2rem;
                }}

                /* Borders */
                .border_bottom {{
                    border-bottom: 2px solid #1E3658;
                }}

                .border_full {{
                    border: 2px solid #1E3658;
                }}

                .border_right {{
                    border-right: 2px solid #1E3658;
                }}

                /* Typography */
                .title{{
                    font-weight: bold;
                    font-size: 1.1rem;
                    color: #1E3658;
                    margin-bottom: 0.5rem;
                }}

                .bold_txt {{
                    font-weight: bold;
                    margin-right: 0.3rem;
                }}

                /* Banner Styles */
                .banner {{
                    width: 100%;
                    height: 45px;
                    background-color: #1E3658;
                    color: white;
                    font-size: 1.5rem;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px 15px;
                    cursor: pointer;
                    user-select: none;
                    position: relative;
                }}

                .consideration .banner {{
                    background-color: #1E3658;
                    margin: 0.8rem 0 0 0;
                }}

                /* Patient Profile */
                .patient_profile {{
                    display: flex;
                    flex-direction: row;
                    gap: 1.5rem;
                    align-items: center;
                    flex: 1;
                }}

                .patient_profile img {{
                    width: 150px;
                    height: 150px;
                    object-fit: cover;
                    border-radius: 8px;
                }}

                .patiant_info {{
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                }}

                /* System Info */
                .system_info {{
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                }}

                .model_info {{
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                }}

                .modality_contrib img {{
                    width: 300px;
                    height: auto;
                }}

                /* Significant Factors */
                .signif_factors {{
                    flex: 1;
                }}

                .signif_items {{
                    display: flex;
                    flex-direction: column;
                    gap: 0.3rem;
                }}

                .signif_item {{
                    display: flex;
                    align-items: flex-start;
                    margin-bottom: 0.5rem;
                    line-height: 1.4;
                }}

                .bullet_point {{
                    display: inline-block;
                    min-width: 8px;
                    height: 8px;
                    background-color: #1E3658;
                    border-radius: 50%;
                    margin-right: 0.5rem;
                    margin-top: 0.4rem;
                }}

                /* Audio Box */
                .audio_box {{
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                }}

                /* Info Box */
                .info_box {{
                    position: relative;
                    padding: 1rem;
                    margin: 1rem 0;
                    width: 100%;
                }}

                .box_title {{
                    position: absolute;
                    top: -15px;
                    left: 15px;
                    color: #1E3658;
                    font-size: 1rem;
                    font-weight: 600;
                    background-color: #E5F1F3;
                    padding: 0 8px;
                    z-index: 3;
                }}

                /* Expandable Content */
                .collapsible-content {{
                    max-height: 0;
                    overflow: hidden;
                    transition: max-height 0.3s ease;
                }}

                .collapsible-content.expanded {{
                    max-height: 2000px;
                }}

                .expand-icon {{
                    margin-left: 10px;
                    transition: transform 0.3s ease;
                    display: inline-block;
                }}

                .expand-icon.expanded {{
                    transform: rotate(90deg);
                }}

                /* Links */
                .more_info a {{
                    color: #1E3658;
                    text-decoration: none;
                    font-style: italic;
                }}

                .more_info a:hover {{
                    text-decoration: underline;
                }}

                /* Consideration Section */
                .consideration p {{
                    width: 100%;
                    text-align: left;
                    padding: 20px;
                    font-weight: 600;
                }}

                /* Child expandable sections */
                .child-banner {{
                    width: 100%;
                    height: 35px;
                    background-color: #5c6879; /* Slightly lighter than parent */
                    color: white;
                    font-size: 1.2rem;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px 15px;
                    cursor: pointer;
                    user-select: none;
                    position: relative;
                    margin-top: 1rem;
                    border-radius: 4px;
                }}
                
                .child-content {{
                    width: 100%;
                    max-height: 0;
                    overflow: hidden;
                    transition: max-height 0.3s ease;
                }}
                
                .child-content.expanded {{
                    max-height: 2000px;
                }}
                
                /* Module specific styles */
                .linguistic-module {{
                    width:100%;
                }}
                
                .acoustic-module {{
                    width:100%;
                }}

                .module-subsection {{
                    width: 100%;
                    margin-bottom: 1rem;
                    padding: 1rem;
                    border: 2px solid #1E3658;
                    border-radius: 4px;
                }}
                
                .module-subsection-title {{
                    font-weight: bold;
                    font-size: 1.1rem;
                    color: #1E3658;
                    margin-bottom: 0.8rem;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                
                .module-subsection-content {{
                    width: 100%;
                }}
                
                /* Adjust existing styles */
                .linguistic-module .info_box {{
                    padding: 0; /* Remove default padding since subsections have their own */
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                }}
                
                .linguistic-module .border_full {{
                    border: none; /* We'll handle borders at subsection level */
                }}

"""
system_prompt_sdh_report = """
You are a clinical language model designed to generate coherent and concise patient summaries based on Social Determinants of Health (SDOH). You will receive:
1) A structured dictionary containing key SDOH items related to a specific patient.
2) A clinical note containing unstructured observations or patient-reported concerns.

Your task is to:
- Read the dictionary and clinical note carefully.
- Write a single, cohesive paragraph that integrates all the information in a natural, readable narrative.
- Use a clinical tone that is clear and factual.
- Do not list items mechanically or use bullet points—integrate them into full sentences.
- Incorporate relevant information from both the structured dictionary and the clinical note.
- **Bold only the negative or high-risk social or behavioral factors** in the summary (e.g., **sedentary lifestyle**).
- Avoid repeating key terms unnecessarily.

Refer to the example below to guide your writing style:

---
## Example Input Dictionary:
{{
    "education": "high school diploma",
    "financial_status": "financial issues",
    "medication_access": "difficulty accessing prescribed medications",
    "smoking": "smokes two packs/day",
    "physical_activity": "sedentary lifestyle",
    "transportation": "transportation barriers to medical services",
    "food_access": "low access to nutritious food (e.g., vegetables)",
    "caregiver": "access to caregivers",
    "health_literacy": "somewhat confident filling out medical forms",
    "insurance": "Medicaid",
    "housing": "stable housing",
    "employment": "unemployed",
    "social_isolation": "often feels isolated",
    "clinical_note": "The patient expressed concern about affording medication and mentioned they have no one to help with groceries or cooking. They also reported difficulty attending regular checkups due to unreliable transportation."
}}
---
## Example Output Report:
Patient has a high school diploma, reports **financial issues**, and is covered by **Medicaid**. They report **difficulty accessing prescribed medications**, **smokes two packs/day**, leads a **sedentary lifestyle**, and faces **transportation barriers to medical services**. There is **low access to nutritious food**, and although the patient reports access to caregivers, they also noted in the clinical note that they **lack support for grocery shopping and cooking**. The patient is **unemployed**, lives in stable housing, and is somewhat confident filling out medical forms. They also **often feel isolated**, which may affect overall well-being.

---

Now generate a similar report for the following input:

---
## Structured SDOH items:
{sdoh_dict}
---
## Clinical Notes:
{clinical notes}
---
## Output Report:
"""



def generate_SDoH_text(SDoH,openai_config):

    prompt = system_prompt_sdh_report.format(sdoh_dict =SDoH )

    model = OpenAI(
            api_key=openai_config['api_key'],
            base_url=openai_config.get('base_url')
        )

    def _generate_openai(prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        return model.chat.completions.create(
            model=kwargs.get('model', "meta-llama/llama-3.1-70b-instruct"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 2048)
        )

    return _generate_openai(prompt).choices[0].message.content



def read_excel_data(excel_path):
    return pd.read_excel(excel_path,index_col=0).to_dict(orient='records')[0]


def generate_interface(
        profile_path:str,
        clinical_factor_path:str,
        lab_tests_path:str,
        model_info_path:str,
        SDoH_path:str,
        openai_config,
        linguistic_interpretation=None,
        transcription=None

    ):
    profile = read_excel_data(profile_path)
    clinical_factor = read_excel_data(clinical_factor_path)
    lab_tests = read_excel_data(lab_tests_path)
    model_info = read_excel_data(model_info_path)
    SDoH = read_excel_data(SDoH_path)

    SDoH = generate_SDoH_text(SDoH,openai_config)
    # print(SDoH)
    significantFactors = [
        "Memory issue manifested by frequent repetition of specific words.",
        "Lack of semantic clarity in speech manifested by reliance on vague terms.",
        "Insomnia and depression manifested as chief complaints.",
        "Low educational attainment manifested by high school diploma.",
    ]

    html = generate_html_report(
        name=profile['name'],
        gender =profile['gender'],
        age=profile['age'],
        cognitive_status=model_info['predicted_status'],
        system_confidence=model_info['confidence'],
        contribution=model_info['contribution'],
        clinical_factor=clinical_factor,
        labTests=lab_tests,
        SDoH=SDoH,
        profileImage="data/profile.png",
        pieChart="data/pie_chart.png",
        audioFile="data/qnvo.mp3",
        significantFactors=significantFactors,
        linguistic_interpretation=linguistic_interpretation,
        transcription=transcription
    )
    return html



def generate_html_report(
    name,
    gender,
    age,
    cognitive_status,
    system_confidence,
    contribution,
    clinical_factor,
    labTests,
    SDoH,
    profileImage="data/profile.png",
    pieChart="data/pie_chart.png",
    audioFile="data/qnvo.mp3",
    significantFactors=None,
    linguistic_interpretation=None,
    transcription=None

):
    
   
    def markdown_bold_to_html(text):
        """
        Converts text with **bold** markdown to HTML with <strong> tags
        Example: "This is **important**" → "This is <strong>important</strong>"
        """
        # Split the text into parts alternating between normal and bold
        parts = text.split('**')
        
        # Rebuild with HTML tags (odd indexes are bold)
        html_content = []
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Odd index = bold section
                html_content.append(f'<strong>{part}</strong>')
            else:
                html_content.append(part)
        
        # Wrap in a div for better HTML structure
        return f'<div class="sdoh-text">{"".join(html_content)}</div>'
    
    # Helper function to format keys
    def format_key(key):
        # Add space before capital letters and numbers
        formatted = re.sub(r'(?<!^)([A-Z])', r' \1', key)        # Space before capital letters, unless at start
        formatted = re.sub(r'(\d+)', r' \1', formatted)          # Space before numbers
        formatted = formatted.title()                            # Capitalize first letter of each word
        formatted = re.sub(r'\s+', ' ', formatted).strip()       # Normalize multiple spaces and trim
        return formatted
    

    SDoH = markdown_bold_to_html(SDoH)
   
    # Generate Patient Status HTML
    patient_status_html = []
    for key, value in clinical_factor.items():
        display_key = key.replace("_", " ").title()  # Simpler alternative to JS formatting
        patient_status_html.append(
            f'<div class="signif_item">'
            f'<span class="bullet_point"></span>'
            f'<span class="bold_txt">{display_key}:</span> {value}'
            f'</div>'
        )

       
    sig_factor_html = []
    for value in significantFactors:
        sig_factor_html.append(
            f'<div class="signif_item">'
            f'<span class="bullet_point"></span>'
            f'{value}'
            f'</div>'
        )

    linguistic_interpretation = [line.strip().split(maxsplit=1)[1]  # Remove first word (bullet symbol)
         for line in linguistic_interpretation.strip().split('\n') 
         if line.strip()]
    ling_interpret_html = []
    for value in linguistic_interpretation:
        ling_interpret_html.append(
            f'<div class="signif_item">'
            f'<span class="bullet_point"></span>'
            f'{value}'
            f'</div>'
        )

    # Generate Lab Tests HTML
    lab_tests_html = []
    for test_name, test_result in labTests.items():
        display_test_name = format_key(test_name)
        lab_tests_html.append(
            f'<div class="signif_item">'
            f'<span class="bullet_point"></span>'
            f'<span class="bold_txt">{display_test_name}:</span> {test_result}'
            f'</div>'
        )

    # Function to encode file to base64
    def encode_file_base64(path):
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    # Encode files
    profile_b64 = encode_file_base64(profileImage)
    piechart_b64 = encode_file_base64(pieChart)
    audio_b64 = encode_file_base64(audioFile)


    return  f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Patient Assessment Report</title>
            <style>
            {css_style}
            </style>
        </head>
        <body>
            <!-- Patient Heading Section -->
            <div class="heading container">
                <div class="vertical_box small_box">
                    <div class="banner"></div>
                    <div class="horizontal_box border_bottom">
                        <div class="patient_profile">
                            <div class="patiant_image">
                                <img src="data:image/png;base64,{profile_b64}" alt="Profile Image">
                            </div>
                            <div class="patiant_info">
                                <h1><span class="bold_txt" style="color: #1E3658;">{name}</span></h1>
                                <p><span class="bold_txt">Gender:</span> <span id="patientGender">{gender}</span></p>
                                <p><span class="bold_txt">Age:</span> <span id="patientAge">{age}</span></p>
                                <p><span class="bold_txt">Primary Language:</span> <span id="patientLanguage">{clinical_factor['primary_language']}</span></p>
                            </div>
                        </div>
                        <div class="system_info">
                            <div class="title">System Outcome</div>
                            <div class="horizontal_box">
                                <div class="model_info">
                                    <div><span class="bold_txt">Cognitive Status:</span><span id="cognitiveStatus">{cognitive_status}</span></div>
                                    <div><span class="bold_txt">System Confidence:</span><span id="systemConfidence">{system_confidence}</span></div>
                                </div>
                                <div class="modality_contrib">
                                    <div class="title">Modality Contribution</div>
                                    <div class="pie_chart">
                                        <img src="data:image/png;base64,{piechart_b64}" alt="Pie chart of the modality contribution">
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="horizontal_box">
                        <div class="signif_factors">
                            <div class="title">Significant Factors</div>
                            <div class="signif_items" id="significantFactors">
                               {"".join(sig_factor_html)}
                            </div>
                            <div class="more_info"><a href="../dashboard/sinificant_features.html">See 20 most important factors ...</a></div>
                        </div>
                        <div class="audio_box">
                            <div class="title">Listen to the audio!</div>
                            <div class="audio_player">
                                <audio controls>
                                    <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                                    Your browser does not support the audio element.
                                </audio>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Clinical Factors and SDoH Section -->
            <div class="clinical_sdoh container">
                <div class="vertical_box small_box">
                    <div class="banner">Clinical Factors and Social Determinants of Health (SDoH) <span class="expand-icon">▼</span></div>
                    <div class="collapsible-content">
                        <div class="info_box border_full horizontal_box">
                            <span class="box_title">Clinical Factors</span>
                            <div class="patien_status border_right">
                                <div class="titlle">Patient's Status:</div>
                                <div class="signif_items" id="patientStatus">
                                   {"".join(patient_status_html)}
                                </div>
                                <div class="more_info"><a href="">More Info...</a></div>
                            </div>
                            <div class="lab_test">
                                <div class="titlle">Lab Tests:</div>
                                <div class="signif_items" id="labTests">
                                    {"".join(lab_tests_html)}
                                </div>
                                <div class="more_info"><a href="">More Info...</a></div>
                            </div>
                        </div>
                        <div class="info_box border_full">
                            <span class="box_title">SDoH</span>
                            <div id="SDoH">{SDoH}</div>
                            <div class="more_info"><a href="">More Info...</a></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Speech Explainability Section -->
            <div class="speech_explainability container">
                <div class="vertical_box small_box">
                    <div class="banner">Speech Explainability <span class="expand-icon">▼</span></div>
                    <div class="collapsible-content">
                        <!-- Linguistic Module -->
                        <div class="linguistic_explainability linguistic-module">
                            <div class="child-banner">
                                Linguistic Module
                                <span class="expand-icon">▼</span>
                            </div>
                            <div class="child-content">
                                <div class="vertical_box">
                                    <div class="info_box">
                                        <!-- Linguistic Interpretation Subsection -->
                                        <div class="module-subsection">
                                            <div class="module-subsection-title">
                                                Linguistic Interpretation
                                                <a href="../dashboard/evidence_linguistic.html">See the evidence</a>
                                            </div>
                                            <div class="module-subsection-content signif_items" id="ling_interpret">
                                                {"".join(ling_interpret_html)}
                                            </div>
                                        </div>
                                        
                                        <!-- Transcription Subsection -->
                                        <div class="module-subsection">
                                            <div class="module-subsection-title">
                                                Transcription
                                            </div>
                                            <div class="module-subsection-content" id="transcription">
                                                {"".join(transcription)}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Acoustic Module -->
                        <div class="acoustic_explainability acoustic-module">
                            <div class="child-banner">
                                Acoustic Module
                                <span class="expand-icon">▲</span>
                            </div>
                            <div class="child-content">
                                <div class="vertical_box">
                                    <div class="info_box border_full">
                                        <a href="../dashboard/evidence_Acoustic.html">See the evidence</a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Consideration Section -->
            <div class="consideration container">
                <div class="vertical_box small_box">
                    <div class="banner">Consideration</div>
                    <p>
                        Please be advised that the sensitivity of this system is not 100%. A more comprehensive
                        evaluation should include the individual's mediacal history and additional cognitive assessments.
                    </p>
                    <div class="info_box border_full">
                        <div>
                            To reduce the risk of cognitive status, evidence-based studies suggested:
                            <div><span class="bullet_point"></span>Having regular excercise</div>
                            <div><span class="bullet_point"></span>Connecting with familty/community</div>
                            <div><span class="bullet_point"></span>Limiting alcohol intake</div>
                        </div>
                    </div>
                </div>
            </div>

            <script>
              
               document.querySelectorAll('.banner').forEach(function(banner) {{
                    const expandIcon = banner.querySelector('.expand-icon');
                    if (!expandIcon) return;

                    const content = banner.parentElement.querySelector('.collapsible-content');

                    // Initialize as collapsed
                    content.classList.remove('expanded');
                    expandIcon.textContent = '▼';
                    expandIcon.classList.remove('expanded');

                    banner.addEventListener('click', function() {{
                    const isExpanded = content.classList.toggle('expanded');
                    expandIcon.classList.toggle('expanded');
                    expandIcon.textContent = isExpanded ? '▲' : '▼';
                    }});
                }});

                
                // Child expandable sections
                document.querySelectorAll('.child-banner').forEach(function(banner) {{
                    const icon = banner.querySelector('.expand-icon');
                    const content = banner.nextElementSibling;
                    
                    // Initialize as collapsed
                    content.classList.remove('expanded');
                    icon.textContent = '▲';
                    
                    banner.addEventListener('click', function(e) {{
                        e.stopPropagation(); // Prevent triggering parent toggle
                        const isExpanded = content.classList.toggle('expanded');
                        icon.textContent = isExpanded ? '▼' : '▲';
                    }});
                }});

            </script>
        </body>
        </html>

    """