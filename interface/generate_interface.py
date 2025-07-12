import re
import base64
import json
from collections import defaultdict
import ast
from typing import Optional, Any, Dict, List

from utils.interface_utils import extract_patient_data,read_excel_sample_data,generate_modality_pie_base64
from interface.LLM_interpretation import generate_SDoH_text,generate_lab_test,generate_clinical_functional,generate_significant_factors

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
            position: relative;

        }}

        .horizontal_box {{
            display: flex;
            width: 100%;
            gap: 5rem;
            justify-content: flex-start;
            align-items: center;
        }}

        /* Borders */
        .border_bottom {{
            border-bottom: 2px solid #1E3658;
        }}

        .border_full {{
            border: 2px solid #1E3658;
            border-radius: 4px;
        }}

        .border_top {{
           border-top: 2px solid #1E3658;
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
        .profile_row{{
            display: flex;
            flex-wrap: nowrap;
            margin-top: 2rem;
        }}

        .sig_factor_row{{
            align-items: center;
            flex-wrap: nowrap;
            margin-top: 2rem;
        }}

        .patient_profile {{
            display: flex;
            gap: 20px;
            align-items: center;
            flex: 1;
        }}

        .patient_image img {{
            width: 240px;
            height: 240px;
            object-fit: cover;
            border-radius: 50%;
        }}

        .patient_info {{
            display: flex;
            flex-direction: column;
            gap: 0px;
            font-size: 1.2rem;
            color: #1E3658;
        }}

        .info_line {{
            font-size: 1.2rem;
            padding:0;
            margin:0;
        }}

        .bold_txt {{
            font-weight: bold;
            margin-right: 6px;
        }}


        .system_outcome_box {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;


        }}

        .system_outcome_box .section_header{{
          margin-bottom:1.5rem;
          border-bottom: 2px solid #7fa37f;
        }}

        .outcome_content {{
            display: flex;
            align-items: flex-start;
            justify-content: flex-start;
            gap: 2rem;
        }}

        .model_info {{
            display: flex;
            flex-direction: column;
            gap: 10px;
            white-space: nowrap;
            flex: 1;
            font-size: 1.2rem;
        }}



        .pie_chart img {{
            width: 260px;
            height: 260px;
        }}

        .pie_chart {{
           display: flex;
           flex-direction: column;
           gap: 2px;
            align-items: center;
            justify-content: center;
        }}

        .pie_chart_title{{
          font-size: 0.9rem;
          font-weight: 200;

        }}

        .pie_chart_info {{
            display: none;
            transition: opacity 0.5s ease-in-out; /* Smooth fade-in effect */
            font-size: 0.9rem;
            margin-top:1rem;
        }}


        .pie_chart_btn {{
            color: #7fa37f;
            font-weight: bold;
            font-size:0.9rem;
            text-decoration: none;
            display: inline-block;
            margin-top: 10px;
            cursor: pointer;
        }}

        .pie_chart_btn:hover {{
            text-decoration: underline;
        }}

        /* Right Side Layout */
        .right_side {{
            flex: 2;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }}



        /* Section Titles */
        .section_header {{
            font-size: 1.4rem;
            color: #1E3658;
            font-weight: bold;
            margin-bottom: 15px;
        }}

        /* System Outcome */
        .system_info {{
            display: flex;
            flex-direction: column;
            gap: 10px;
            font-size: 1.2rem;
        }}

        /* Modality Contribution */
        .modality_contrib {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }}


        .signif_factors {{
            flex: 1;
        }}

        .accent_color{{

            color:#350a29 !important;
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
            width: 8px !important;
            height: 8px !important;
            background-color: #1E3658;
            border-radius: 50%;
            margin-right: 8px;
            margin-top: 7px;
            flex-shrink: 0;
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
            font-size: 1.1rem;
            font-weight: 600;
            background-color: #E5F1F3;
            padding: 0 8px;
        }}

        .acoustic_iterpretation {{
            margin: 1rem;

        }}

        /* Expandable Content */
        .collapsible-content {{
            width: 100%;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}

        .collapsible-content.expanded {{
            max-height: 2000px;
            width: 100%;
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
        .consideration-section {{
            position: relative; /* Enables absolute positioning inside it */
        }}

        .consideration-section.banner{{
          margin-bottom:2rem;
        }}

        .right-image {{
            position:absolute;
            bottom:0;
            right:0;
            width: 400px;
            height: auto;
            opacity: 0.5; /* controls transparency */
            pointer-events: none; /* makes sure it doesn’t block clicks */
            z-index:-1;
            transform: translateY(20%)

        }}

        .footer{{
          margin-top:2rem;
        }}

        .footer > *{{
          margin-top:1.2rem;
        }}

        .footer> p{{
          width: 65% !important;
        }}
        .footer > .border_full{{
          width: 65%!important;
        }}

        /* Child expandable sections */
        .child-banner {{
            width: 100%;
            height: 40px;
            background-color: #7fa37f;
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

        .module-subsection-content .signif_items .signif_item {{
            display: flex;
            align-items: flex-start;
            margin-bottom: 0.5rem;
            line-height: 1.4;
        }}

        .module-subsection-content .bullet_point {{
            display: inline-block;
            min-width: 6px;  /* Reduced from 8px */
            height: 6px;     /* Reduced from 8px */
            background-color: #1E3658;
            border-radius: 50%;
            margin-right: 8px;
            margin-top: 7px;  /* Adjusted for better vertical alignment */
            flex-shrink: 0;   /* Prevents bullet from shrinking */
        }}

        .module-subsection-content.vertical_box {{
            display: flex;
            flex-direction: column;
            gap: 0;
        }}

        .image-block {{
            width: 100%;
            position: relative;
        }}

        .image-block img {{
            width: 100%;
            display: block;
            object-fit: cover;
        }}

        /* Subtitles */
        .subtitle {{
            background-color: white;
            margin: 0;
            padding: 6px 10px;
            text-align: center;
            font-size: 1.8 rem;
            width: 100%;
            box-sizing: border-box;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border-top: 1px solid #ccc;
            color: #1E3658;
        }}
        .category-header {{
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 5px;
            font-size: 1.1em;
            color:#004d0c;
        }}
        .item-key {{
            font-weight: 600;
            margin-right: 5px;
        }}

"""





def generate_interface(
    excel_path: str,
    model_info: Any,
    openai_config: Dict,
    profile_image_path: str,
    audio_path: str,
    linguistic_interpretation: Optional[str] = None,
    patient_id: Optional[str] = None,
    transcription: Optional[str] = None,
    acoustic_plots: Optional[Any] = None,
    acoustic_interpretation: Optional[str] = None,
    max_retries: int = 3
) -> Dict:
    """
    Generate a patient interface with various data components.
    
    Args:
        excel_path: Path to the Excel file containing patient data
        model_info: Model information
        openai_config: Configuration for OpenAI API
        profile_image_path: Path to the patient's profile image
        audio_path: Path to the patient's audio file
        linguistic_interpretation: Linguistic interpretation data
        patient_id: ID of the patient
        transcription: Transcription of the audio
        acoustic_plots: Acoustic plots data
        acoustic_interpretation: Acoustic interpretation data
        max_retries: Maximum number of retries for parsing operations
        
    Returns:
        Dictionary containing all the generated interface data
    """
    # Helper functions with retry logic
    def extract_list_from_string(text: str, retries: int = max_retries) -> Optional[List[Dict]]:
        """
        Extract a list from a string with retry logic.
        
        Args:
            text: String containing potential list data
            retries: Number of remaining retry attempts
            
        Returns:
            Parsed list or None if unsuccessful after all retries
        """
        for attempt in range(retries):
            try:
                match = re.search(r'\[\s*\{.*\}\s*\]', text)
                if not match:
                    continue
                
                list_str = match.group(0)
                return json.loads(list_str)
            except (json.JSONDecodeError, AttributeError):
                if attempt == retries - 1:
                    return None
                continue
        return None

    def parse_json_data(json_str: str, retries: int = max_retries) -> Optional[Dict]:
        """
        Parse JSON data with retry logic.
        
        Args:
            json_str: String containing JSON data
            retries: Number of remaining retry attempts
            
        Returns:
            Parsed dictionary or None if unsuccessful after all retries
        """
        for attempt in range(retries):
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                if attempt == retries - 1:
                    return None
                continue
        return None

    def parse_ast_literal(literal_str: str, retries: int = max_retries) -> Optional[Any]:
        """
        Parse a string using ast.literal_eval with retry logic.
        
        Args:
            literal_str: String to evaluate
            retries: Number of remaining retry attempts
            
        Returns:
            Parsed Python object or None if unsuccessful after all retries
        """
        for attempt in range(retries):
            try:
                return ast.literal_eval(literal_str)
            except (ValueError, SyntaxError):
                if attempt == retries - 1:
                    return None
                continue
        return None

    # Main processing
    data = extract_patient_data(read_excel_sample_data(excel_path, patient_id))
    profile = data["Demographic and Basic Information"]
    
    # Generate SDoH
    SDoH = generate_SDoH_text(
        data['Social Determinants of Health (SDoH)'],
        data['Clinical Note']['Report'],
        openai_config
    )
    
    # Process lab tests with retry
    lab_tests_str = generate_lab_test(data['Laboratory Tests & Biomarkers'], openai_config)
    lab_tests = extract_list_from_string(lab_tests_str)
    
    # Process clinical functional data with retry
    clinical_functional_dict = {
        key: data[key] for key in [
            "Demographic and Basic Information",
            "Psychological and Behavioral",
            "Functional Status",
            "Cognitive Symptoms",
            "Physiological",
            "Medical Interventions and Therapies"
        ]
    }
    clinical_functional_str = generate_clinical_functional(clinical_functional_dict, openai_config)
    clinical_functional = parse_json_data(clinical_functional_str)
    
    # Process significant factors with retry
    significant_factors_str = generate_significant_factors(
        SDoH,
        clinical_functional_str,
        linguistic_interpretation,
        acoustic_interpretation,
        openai_config
    )
    significantFactors = parse_ast_literal(significant_factors_str)

    return generate_html_report(
        name=profile['Full Name'],
        gender =profile['Gender'],
        age=profile['Age'],
        language = profile['Primary Language'],
        cognitive_status=model_info['predicted_status'],
        system_confidence=model_info['confidence'],
        contribution=model_info['contribution'],
        clinical_functional_factor=clinical_functional,
        lab_tests=lab_tests,
        SDoH=SDoH,
        profileImage=profile_image_path,
        audioFile=audio_path,
        acoustic_plots=acoustic_plots,
        significantFactors=significantFactors,
        linguistic_interpretation=linguistic_interpretation,
        acoustic_interpretation = acoustic_interpretation,
        transcription=transcription
    )


def generate_html_report(
    name,
    gender,
    age,
    language,
    cognitive_status,
    system_confidence,
    contribution,
    clinical_functional_factor,
    lab_tests,
    SDoH,
    acoustic_plots,
    profileImage="data/profile.jpeg",
    decorationImage = "data/decoration.png",
    audioFile="data/qnvo.mp3",
    significantFactors=None,
    linguistic_interpretation=None,
    acoustic_interpretation =None,
    transcription=None

):
    def format_linguistic_interpretation_html(text, title_color="#1E3658", prediction_color="#1E3658"):

        lines = text.strip().split('\n')


        bullet_symbols = ('*', '•')
        bullet_start_index = next(
            (i for i, line in enumerate(lines) if line.strip().startswith(bullet_symbols)),
            0
        )
        relevant_lines = lines[bullet_start_index:]

        html_output = []

        for line in relevant_lines:
            bullet_symbols = ('*', '•')
            line = line.strip().lstrip(''.join(bullet_symbols)).strip()

            if not line:
                continue

            if line.startswith("The speaker is"):
                html_output.append(
                    f'<div class="signif_item border_top">'
                    f'<b style="color:{prediction_color}">{line}</b>'
                    f'</div>'
                )
            else:
                try:
                    title_part, body_part = line.split(':', maxsplit=1)
                    title = title_part.replace("**", "").strip()
                    body = body_part.strip().replace("**", "")
                    html_output.append(
                        f'<div class="signif_item">'
                        f'<span class="bullet_point"></span>'
                        f'<b style="color:{title_color}; white-space:nowrap">{title}:</b> {body}'
                        f'</div>'
                    )
                except ValueError:
                    # If there's a formatting issue
                    html_output.append(
                        f'<div class="signif_item">{line}</div>'
                    )

        return html_output

    def process_clinical_data(data):
        """Process the new format clinical data into grouped HTML"""
  
        # Group items by category
        category_items = defaultdict(list)
        for item in data:
            for key, (value, category) in item.items():
                category_items[category].append((key, value))

        # Generate HTML
        html_elements = []
        for category, items in category_items.items():
            # Add category header
            html_elements.append(f'<div class="category-header">{category}</div>')

            # Add all items for this category
            for key, value in items:
                html_elements.append(
                    f'<div class="signif_item">'
                    f'<div class="bullet_point"></div>'
                    f'<span class="item-key">{key}:</span> '
                    f'<span class="item-value">{value}</span>'
                    f'</div>'
                )

        return '\n'.join(html_elements)

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
    clinical_behavioral = process_clinical_data(clinical_functional_factor)
    ling_interpret_html = format_linguistic_interpretation_html(linguistic_interpretation)

    sig_factor_html = []
    for value in significantFactors:
        sig_factor_html.append(
            f'<div class="signif_item">'
            f'<span class="bullet_point"></span>'
            f'{value}'
            f'</div>'
        )

    spect_interpret_html = []
    for key, line in acoustic_interpretation.items():
      if key in ['pause', 'energy', 'f0', 'f3']:
          spect_interpret_html.append(
              f'<div class="signif_item">'
              f'  <span class="bullet_point"></span>'
              f'  {line}'
              f'</div>'
          )

    entropy_interpret_html = []
    for key, line in acoustic_interpretation.items():
      if key in ['entropy']:
          entropy_interpret_html.append(
              f'<div class="signif_item">'
              f'  <span class="bullet_point"></span>'
              f'  {line}'
              f'</div>'
          )

    saliency_interpret_html = []
    for key, line in acoustic_interpretation.items():
      if key in ['shimmer']:
          saliency_interpret_html.append(
              f'<div class="signif_item">'
              f'  <span class="bullet_point"></span>'
              f'  {line}'
              f'</div>'
          )

    lab_tests_html = []
    for item in lab_tests:
        for key, value in item.items():
            display_test_name = format_key(key)
            lab_tests_html.append(
                f'<div class="signif_item">'
                f'<span class="bullet_point"></span>'
                f'<span class="item-key">{display_test_name}:</span> {value}'
                f'</div>'
            )

    # Function to encode file to base64
    def encode_file_base64(path):
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    # Encode files
    profile_b64 = encode_file_base64(profileImage)
    piechart_b64 = generate_modality_pie_base64(contribution)
    waveform_b64 = acoustic_plots['colored_waveform']
    shap_spect_b64 = acoustic_plots['SHAP_highlighted_spectrogram']
    entropy_b64 = acoustic_plots['entropy']
    decoration_b64 = encode_file_base64(decorationImage)
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
                    <div class="vertical_box border_bottom">
                      <div class="horizontal_box profile_row">
                        <!-- Patient Profile -->
                        <div class="patient_profile">
                          <div class="patient_image">
                            <img src="data:image/png;base64,{profile_b64}" alt="Profile Image">
                          </div>
                          <div class="patient_info">
                            <div class="info_line"><span class="bold_txt">Name:</span> <span>{name}</span></div>
                            <div class="info_line"><span class="bold_txt">Gender:</span> <span id="patientGender">{gender}</span></div>
                            <div class="info_line"><span class="bold_txt">Age:</span> <span id="patientAge">{age}</span></div>
                            <div class="info_line"><span class="bold_txt">Primary Language:</span> <span id="patientLanguage">{language}</span></div>
                          </div>
                        </div>

                        <!-- System Outcome Section -->
                        <div class="system_outcome_box">
                          <div class="section_header">System Outcome</div>
                          <div class="horizontal_box outcome_content">
                            <div class="model_info">
                              <div><span class="bold_txt accent_color">Cognitive Status:</span> <span id="cognitiveStatus" class="bold_txt">{cognitive_status.upper()}</span></div>
                              <div><span class="bold_txt accent_color">System Confidence:</span> <span id="systemConfidence" class="bold_txt">{float(system_confidence)*100} %</span></div>
                            </div>
                            <div class="pie_chart">

                              <img src="data:image/png;base64,{piechart_b64}" alt="Pie chart of the modality contribution">
                              <div><span class="bold_txt pie_chart_title">Modality Contribution of Speech & Demographic </span><span class="pie_chart_btn" >(More Info)</span></div>

                            </div>
                          </div>
                        </div>
                      </div>
                      <div class="pie_chart_info">
                        ** The baseline SpeechCARE model was trained using only speech and demographic information. At this stage, we are unable to generate a pie chart showing
                        the separate contributions of Clinical, Speech, and Demographic information.
                        However, as we enhance the model by incorporating clinical, demographic, and speech data, the updated pie chart will reflect the contribution of each modality more comprehensively.
                      </div>
                    </div>


                    <!-- Second Row -->
                    <div class="horizontal_box sig_factor_row">
                        <div class="signif_factors">
                            <div class="title">Significant Factors</div>
                            <div class="signif_items" id="significantFactors">
                                {"".join(sig_factor_html)}
                            </div>
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
                    <div class="banner">Patient Health Assessment <span class="expand-icon">▼</span></div>
                    <div class="collapsible-content">
                        <div class="info_box border_full">
                            <span class="box_title bold_txt">Clinical and Functional Overview</span>
                            <div id="clinical_behavioral">{clinical_behavioral}</div>
                        </div>
                        <div class="info_box border_full">
                            <span class="box_title bold_txt">Lab Tests</span>
                            <div id="lab_tests">{"".join(lab_tests_html)}</div>
                        </div>
                        <div class="info_box border_full">
                            <span class="box_title bold_txt">Social Determinants of Health (SDOH)</span>
                            <div id="SDoH">{SDoH}</div>
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
                                        <div class="module-subsection-content" id="transcription">
                                            {"".join(transcription)}
                                        </div>
                                    </div>
                                    <!-- Linguistic Interpretation Subsection -->
                                    <div class="module-subsection">

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
                                <span class="expand-icon">▼</span>
                            </div>
                            <div class="child-content">
                                <div class="module-subsection">
                                    <div class="module-subsection-content vertical_box">
                                    <div class="image-block waveform">
                                        <img src="data:image/png;base64,{waveform_b64}" alt="Waveform">
                                        <div class=" signif_items acoustic_iterpretation">
                                            {"".join(saliency_interpret_html)}
                                        </div>
                                    </div>
                                    <div class="image-block spectrogram">
                                        <img src="data:image/png;base64,{shap_spect_b64}" alt="SHAP Spectrogram">
                                        <div class=" signif_items acoustic_iterpretation">
                                            {"".join(spect_interpret_html)}
                                        </div>
                                    </div>
                                    <div class="image-block entropy">
                                        <img src="data:image/png;base64,{entropy_b64}" alt="Entropy">
                                        <div class="signif_items acoustic_iterpretation">
                                            {"".join(entropy_interpret_html)}
                                        </div>
                                    </div>
                                    </div>
                                </div>

                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Consideration Section -->
            <div class="container consideration-section">
                <div class="vertical_box">
                    <div class="banner">Consideration</div>
                    <div class="footer">
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
                      <img src="data:image/png;base64,{decoration_b64}" alt="Decoration" class="right-image">
                    </div>
                </div>
            </div>




            <script>

                // Wait a moment for elements to exist
                setTimeout(() => {{
                  const btn = document.querySelector('.pie_chart_btn');
                  const info = document.querySelector('.pie_chart_info');

                  btn.addEventListener('click', () => {{
                    if (info.style.display === 'none') {{
                      info.style.display = 'block';
                      btn.textContent = '(Less Info)';
                    }} else {{
                      info.style.display = 'none';
                      btn.textContent = '(More Info)';
                    }}
                  }});
                }}, 500); // Small delay to ensure DOM is ready
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
                    icon.textContent = '▼';

                    banner.addEventListener('click', function(e) {{
                        e.stopPropagation();
                        const isExpanded = content.classList.toggle('expanded');
                        icon.textContent = isExpanded ? '▲' : '▼';
                    }});
                }});
            </script>
        </body>
        </html>

    """