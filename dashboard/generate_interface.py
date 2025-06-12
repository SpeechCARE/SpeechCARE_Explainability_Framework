
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
            font-size: 16px;
            line-height: 1.5;
            color: #1E3658;
            padding: 20px 0;
        }}
        
        /* Layout Components */
        .container {{
            width: 100%;
            max-width: 1200px;
            margin: 0 auto 20px;
        }}

        .vertical_box {{
            display: flex;
            flex-direction: column;
            width: 100%;
        }}

        .horizontal_box {{
            display: flex;
            width: 100%;
            gap: 30px;
            padding: 15px 0;
        }}

        /* Borders */
        .border_bottom {{
            border-bottom: 2px solid #1E3658;
            padding-bottom: 15px;
        }}

        .border_full {{
            border: 2px solid #1E3658;
            border-radius: 4px;
        }}

        .border_right {{
            border-right: 2px solid #1E3658;
            padding-right: 15px;
        }}

        /* Typography */
        .title {{
            font-weight: bold;
            font-size: 1.1rem;
            color: #1E3658;
            margin-bottom: 10px;
        }}

        .bold_txt {{
            font-weight: bold;
            margin-right: 5px;
        }}

        /* Banner Styles */
        .banner {{
            width: 100%;
            height: 45px;
            background-color: #1E3658;
            color: white;
            font-size: 1.2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 15px;
            cursor: pointer;
            user-select: none;
            border-radius: 4px 4px 0 0;
        }}

        /* Patient Profile */
        .patient_profile {{
            display: flex;
            gap: 20px;
            align-items: center;
            flex: 1;
        }}

        .patient_profile img {{
            width: 120px;
            height: 120px;
            object-fit: cover;
            border-radius: 8px;
        }}

        .patiant_info {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        /* System Info */
        .system_info {{
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}

        .model_info {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .modality_contrib img {{
            width: 100%;
            max-width: 300px;
            height: auto;
        }}

        /* Significant Factors */
        .signif_factors {{
            flex: 1;
        }}

        .signif_items {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .signif_item {{
            display: flex;
            align-items: flex-start;
            line-height: 1.4;
        }}

        .bullet_point {{
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #1E3658;
            border-radius: 50%;
            margin-right: 8px;
            margin-top: 7px;
        }}

        /* Audio Box */
        .audio_box {{
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        /* Info Box */
        .info_box {{
            position: relative;
            padding: 20px;
            margin: 20px 0;
            width: 100%;
        }}

        .box_title {{
            position: absolute;
            top: -12px;
            left: 15px;
            color: #1E3658;
            font-size: 1rem;
            font-weight: 600;
            background-color: #E5F1F3;
            padding: 0 8px;
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
        }}

        /* Links */
        .more_info a {{
            color: #1E3658;
            text-decoration: none;
            font-style: italic;
            display: inline-block;
            margin-top: 10px;
        }}

        .more_info a:hover {{
            text-decoration: underline;
        }}

        /* Consideration Section */
        .consideration p {{
            padding: 15px 0;
            font-weight: 600;
        }}

        /* Child expandable sections */
        .child-banner {{
            width: 100%;
            height: 40px;
            background-color: #5c6879;
            color: white;
            font-size: 1.1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 15px;
            cursor: pointer;
            user-select: none;
            margin-top: 15px;
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
        .module-subsection {{
            width: 100%;
            margin-bottom: 15px;
            padding: 15px;
            border: 2px solid #1E3658;
            border-radius: 4px;
        }}
        
        .module-subsection-title {{
            font-weight: bold;
            font-size: 1.1rem;
            color: #1E3658;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        /* Audio player styling */
        audio {{
            width: 100%;
            max-width: 400px;
        }}

"""
system_prompt_sdh_report = f"""
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
{{sdoh_dict}}
---
## Clinical Notes:
{{clinical_notes}}
---
## Output Report:
"""


def generate_SDoH_text(SDoH,clinical_notes,openai_config):

    data_ = {}
    for key,value in SDoH.items():
        data_[key.split(":")[0]] = value

    prompt = system_prompt_sdh_report.format(sdoh_dict =data_,clinical_notes=clinical_notes )

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


def extract_patient_data(df):
    """
    Extracts structured sections from a two-column DataFrame. Handles missing 'Clinical Notes' gracefully.

    Args:
        df (pd.DataFrame): A dataframe with two columns: [Section, Value]

    Returns:
        dict: A dictionary with structured patient data categorized by section.
    """
    df.columns = ['Section', 'Value']

    # Define section markers
    sections = {
        "Demographic Information": "Demographic information",
        "Clinical History": "Clinical History",
        "Social Determinants of Health": "Social of determinant of Health",
        "Lab Tests": "Lab Tests",
        "Clinical Notes": "Clinical Notes"
    }

    # Detect where each section begins (only include if present)
    section_indices = {
        key: df[df['Section'] == label].index[0]
        for key, label in sections.items()
        if label in df['Section'].values
    }

    # Sort by row index to determine boundaries
    sorted_sections = sorted(section_indices.items(), key=lambda x: x[1])
    boundaries = {
        key: (idx, sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(df))
        for i, (key, idx) in enumerate(sorted_sections)
    }

    # Extract data for each section
    structured_data = {}
    for section, (start, end) in boundaries.items():
        sub_df = df.iloc[start + 1:end].dropna(subset=['Section'])
        structured_data[section] = {
            str(row['Section']).strip(): str(row['Value']).strip()
            for _, row in sub_df.iterrows()
            if str(row['Section']).strip().lower() != 'nan'
        }

    return structured_data

def read_excel_data(excel_path,id=None):
    xls = pd.ExcelFile(excel_path, engine='openpyxl')
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

    new_df = pd.DataFrame({
    'Section': ['Demographic information'] + df['Demographic information'].tolist(),
    'Value': ['NaN'] + df[df.columns[1]].tolist()
    })

    return new_df


def generate_interface(
        excel_path:str,
        model_info_path:str,
        openai_config,
        linguistic_interpretation=None,
        patient_id:str=None,
        transcription=None

    ):
    data = extract_patient_data(read_excel_data(excel_path,patient_id))
    model_info = read_excel_data(model_info_path)

    SDoH = generate_SDoH_text(data['Social Determinants of Health'],data['Clinical Notes']['Report:'],openai_config)
    profile = data['Demographic Information']
    lab_tests = data['Lab Tests']
    clinical_factor = data['Clinical History']
    significantFactors = [
        "Memory issue manifested by frequent repetition of specific words.",
        "Lack of semantic clarity in speech manifested by reliance on vague terms.",
        "Insomnia and depression manifested as chief complaints.",
        "Low educational attainment manifested by high school diploma.",
    ]

    html = generate_html_report(
        name=profile['Name'],
        gender =profile['Gender'],
        age=profile['Age'],
        language = profile['Primary Language'],
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
    language,
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
        if key.isupper():
            return key
        
        # Add space before capital letters and numbers
        
        formatted = re.sub(r'(?<!^)([A-Z])', r' \1', key)        # Space before capital letters, unless at start
        # formatted = re.sub(r'(\d+)', r' \1', key)          # Space before numbers
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
            <div class="container">
                <div class="vertical_box">
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
                                <p><span class="bold_txt">Primary Language:</span> <span id="patientLanguage">{language}</span></p>
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
            <div class="container">
                <div class="vertical_box">
                    <div class="banner">Clinical Factors and Social Determinants of Health (SDoH) <span class="expand-icon">▼</span></div>
                    <div class="collapsible-content">
                        <div class="info_box border_full horizontal_box">
                            <span class="box_title">Clinical Factors</span>
                            <div class="patien_status border_right">
                                <div class="title">Patient's Status:</div>
                                <div class="signif_items" id="patientStatus">
                                    {"".join(patient_status_html)}
                                </div>
                                <div class="more_info"><a href="">More Info...</a></div>
                            </div>
                            <div class="lab_test">
                                <div class="title">Lab Tests:</div>
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
            <div class="container">
                <div class="vertical_box">
                    <div class="banner">Speech Explainability <span class="expand-icon">▼</span></div>
                    <div class="collapsible-content">
                        <!-- Linguistic Module -->
                        <div class="linguistic-module">
                            <div class="child-banner">
                                Linguistic Module
                                <span class="expand-icon">▼</span>
                            </div>
                            <div class="child-content">
                                <div class="info_box">
                                    <!-- Transcription Subsection -->
                                    <div class="module-subsection">
                                        <div class="module-subsection-title">
                                            Transcription
                                        </div>
                                        <div class="module-subsection-content" id="transcription">
                                            {"".join(transcription)}
                                        </div>
                                    </div>
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
                                </div>
                            </div>
                        </div>
                        
                        <!-- Acoustic Module -->
                        <div class="acoustic-module">
                            <div class="child-banner">
                                Acoustic Module
                                <span class="expand-icon">▲</span>
                            </div>
                            <div class="child-content">
                                <div class="info_box border_full">
                                    <a href="../dashboard/evidence_Acoustic.html">See the evidence</a>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Consideration Section -->
            <div class="container">
                <div class="vertical_box">
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

                    const content = banner.nextElementSibling;

                    // Initialize as collapsed
                    content.classList.remove('expanded');
                    expandIcon.textContent = '▼';

                    banner.addEventListener('click', function() {{
                        const isExpanded = content.classList.toggle('expanded');
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
                        e.stopPropagation();
                        const isExpanded = content.classList.toggle('expanded');
                        icon.textContent = isExpanded ? '▼' : '▲';
                    }});
                }});
            </script>
        </body>
        </html>

    """