
import pandas as pd
import OpenAI

system_prompt_sdh_report = """
    You are a clinical language model designed to generate coherent and concise patient summaries based on Social Determinants of Health (SDOH). You will receive:
    1) A structured dictionary containing key SDOH items related to a specific patient.

    Your task is to:
    - Read the dictionary carefully.
    - Write a single, cohesive paragraph that integrates all the information in a natural, readable narrative.
    - Use a clinical tone that is clear and factual.
    - Do not list items mechanically or use bullet points—integrate them into full sentences.
    - Avoid repeating key terms unnecessarily.

    Refer to the example below to guide your writing style:

    ---
    ## Example Input Dictionary:
    {
        "education": "high school diploma",
        "financial_status": "financial issues",
        "medication_access": "difficulty accessing prescribed medications",
        "smoking": "smokes two packs/day",
        "physical_activity": "sedentary lifestyle",
        "transportation": "transportation barriers to medical services",
        "food_access": "low access to nutritious food (e.g., vegetables)",
        "caregiving": "access to caregivers"
    }
    ---
    ## Example Output Report:
    Patient has a high school diploma, reports financial issues, difficulty accessing prescribed medications, smokes two packs/day, leads a sedentary lifestyle, faces transportation barriers to medical services, has low access to nutritious food (e.g., vegetables), and reports access to caregivers.
    ---

    Now generate a similar report for the following input:

    ---
    ## Input Dictionary:
    {sdoh_dict}
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
    return pd.read_excel(excel_path).to_dict(orient='records')[0]
    

def generate_interface(
        profile_path:str,
        clinical_factor_path:str,
        lab_tests_path:str,
        model_info_path:str,
        SDoH_path:str,
        openai_config,
        
    ):
    profile = read_excel_data(profile_path)
    clinical_factor = read_excel_data(clinical_factor_path)
    lab_tests = read_excel_data(lab_tests_path)
    model_info = read_excel_data(model_info_path)
    SDoH = read_excel_data(SDoH_path)

    SDoH = generate_SDoH_text(SDoH,openai_config)
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
        profileImage="../data/profile.png",
        pieChart="../data/pie_chart.png",
        audioFile="../data/qnvo.mp3",
        significantFactors=significantFactors,
    )

    

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
    profileImage="../data/profile.png",
    pieChart="../data/pie_chart.png",
    audioFile="../data/qnvo.mp3",
    significantFactors=None,
   
):
    # Convert data to JavaScript object format
    def format_js_obj(obj, indent=8):
        if isinstance(obj, dict):
            items = []
            for k, v in obj.items():
                if isinstance(v, str):
                    items.append(f'"{k}": "{v}"')
                else:
                    items.append(f'"{k}": {format_js_obj(v, indent+4)}')
            return "{\n" + " "*(indent+4) + ",\n".join(items) + "\n" + " "*indent + "}"
        elif isinstance(obj, list):
            items = [f'"{item}"' if isinstance(item, str) else str(item) for item in obj]
            return "[" + ", ".join(items) + "]"
        else:
            return str(obj)

    patientStatus_js = format_js_obj(clinical_factor)
    labTests_js = format_js_obj(labTests)
    significantFactors_js = format_js_obj(significantFactors)
    html_template = f"""

<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Patient Assessment Report</title>
    <style>
        /* Reset and Base Styles */
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
            color: #333;
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
            width: 80%;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .horizontal_box {{
            width: 100%;
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            padding: 1rem;
            gap: 1rem;
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
            background-color: #5c6879;
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
            width: 120px;
            height: 120px;
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
            width: 250px;
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
    </style>
</head>
<body>
    <!-- Patient Heading Section -->
    <div class="heading container">
        <div class="vertical_box">
            <div class="banner"></div>
            <div class="horizontal_box border_bottom">
                <div class="patient_profile">
                    <div class="patiant_image">
                        <img id="profileImage" src="" alt="Patient image">
                    </div>
                    <div class="patiant_info">
                        <h1 id="patientName"></h1>
                        <p><span class="bold_txt">Gender:</span> <span id="patientGender"></span></p>
                        <p><span class="bold_txt">Age:</span> <span id="patientAge"></span></p>
                        <p><span class="bold_txt">Primary Language:</span> <span id="patientLanguage"></span></p>
                    </div>
                </div>
                <div class="system_info">
                    <div class="title">System Outcome</div>
                    <div class="horizontal_box">                   
                        <div class="model_info">
                            <div><span class="bold_txt">Cognitive Status:</span><span id="cognitiveStatus"></span></div>
                            <div><span class="bold_txt">System Confidence:</span><span id="systemConfidence"></span></div>
                        </div>
                        <div class="modality_contrib">
                            <div class="title">Modality Contribution</div>
                            <div class="pie_chart">
                                <img id="pieChart" src="" alt="Pie chart of the modality contribution">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="horizontal_box">
                <div class="signif_factors">
                    <div class="title">Significant Factors</div>
                    <div class="signif_items" id="significantFactors">
                        <!-- Will be filled by JavaScript -->
                    </div>
                    <div class="more_info"><a href="../dashboard/sinificant_features.html">See 20 most important factors ...</a></div>
                </div>
                <div class="audio_box">
                    <div class="title">Listen to the audio!</div>
                    <div class="audio_player">
                        <audio controls autoplay>
                            <source id="audioSource" src="../data/qnvo.mp3" type="audio/mpeg">
                        </audio>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Clinical Factors and SDoH Section -->
    <div class="clinical_sdoh container">
        <div class="vertical_box">
            <div class="banner">Clinical Factors and Social Determinants of Health (SDoH) <span class="expand-icon">▼</span></div>
            <div class="collapsible-content">
                <div class="info_box border_full horizontal_box">
                    <span class="box_title">Clinical Factors</span>
                    <div class="patien_status border_right">
                        <div class="titlle">Patient's Status:</div>
                        <div class="signif_items" id="patientStatus">
                            <!-- Will be filled by JavaScript -->
                        </div>
                        <div class="more_info"><a href="">More Info...</a></div>
                    </div>
                    <div class="lab_test">
                        <div class="titlle">Lab Tests:</div>
                        <div class="signif_items" id="labTests">
                            <!-- Will be filled by JavaScript -->
                        </div>
                        <div class="more_info"><a href="">More Info...</a></div>
                    </div>
                </div>
                <div class="info_box border_full">
                    <span class="box_title">SDoH</span>
                    <div id="SDoH"></div>
                    <div class="more_info"><a href="">More Info...</a></div>    
                </div>
            </div>
        </div>
    </div>
    
    <!-- Speech Explainability Section -->
    <div class="speech_explainability container">
        <div class="vertical_box">
            <div class="banner">Speech Explainability <span class="expand-icon">▼</span></div>
            <div class="collapsible-content">
                <div class="linguistic_explainability">
                    <div class="vertical_box">
                        <div class="info_box border_full">
                            <span class="box_title">Linguistic Module</span>
                            <a href="../dashboard/evidence_linguistic.html">See the evidence</a>
                        </div>
                    </div>
                </div>
                <div class="acoustic_explainability">
                    <div class="vertical_box">
                        <div class="info_box border_full">
                            <span class="box_title">Acoustic Module</span>
                            <a href="../dashboard/evidence_Acoustic.html">See the evidence</a>
                        </div>
                    </div>
                </div>
            </div>  
        </div>
    </div>
    
    <!-- Consideration Section -->
    <div class="consideration container">
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
        // Sample data - replace with your actual data retrieval logic
        const data = {{
            name: "{name}",
            gender: "{gender}",
            age: "{age}",
            language: "{clinical_factor['language']}",
            cognitive_status: "{cognitive_status}",
            system_confidence: "{system_confidence}",
            profileImage: "{profileImage}",
            pieChart: "{pieChart}",
            audioFile: "{audioFile}",
            significantFactors: {significantFactors_js},
            patientStatus: {patientStatus_js},
            labTests: {labTests_js},
            SDoH: "{SDoH}"
        }};

         // Function to populate the HTML with data
        function populateData() {{
            // Patient Info
            document.getElementById('patientName').textContent = data.name;
            document.getElementById('patientGender').textContent = data.gender;
            document.getElementById('patientAge').textContent = data.age;
            document.getElementById('patientLanguage').textContent = data.language;
            
            // System Info
            document.getElementById('cognitiveStatus').textContent = data.cognitive_status;
            document.getElementById('systemConfidence').textContent = data.system_confidence;
            
            // Images and Audio
            document.getElementById('profileImage').src = data.profileImage;
            document.getElementById('pieChart').src = data.pieChart;
            document.getElementById('audioSource').src = data.audioFile;
            
            // SDOH
            document.getElementById('SDoH').textContent = data.SDoH;
            
            // Significant Factors
            const factorsContainer = document.getElementById('significantFactors');
            data.significantFactors.forEach(factor => {{
                const factorElement = document.createElement('div');
                factorElement.className = 'signif_item';
                factorElement.innerHTML = `<span class="bullet_point"></span>${{factor}}`;
                factorsContainer.appendChild(factorElement);
            }});

            // Patient Status
            const statusContainer = document.getElementById('patientStatus');
            for (const [key, value] of Object.entries(data.patientStatus)) {{
                // Convert key to display format (e.g., "primaryDiagnosis" -> "Primary Diagnosis")
                const displayKey = key.replace(/([A-Z])/g, ' $1') // Add space before capitals
                                    .replace(/^./, str => str.toUpperCase()); // Capitalize first letter
                
                const itemElement = document.createElement('div');
                itemElement.className = 'signif_item';
                itemElement.innerHTML = `
                    <span class="bullet_point"></span>
                    <span class="bold_txt">${{displayKey}}:</span> ${{value}}
                `;
                statusContainer.appendChild(itemElement);
            }};
            // Lab Tests
            const labTestsContainer = document.getElementById('labTests');
            for (const [testName, testResult] of Object.entries(data.labTests)) {{
                // Convert testName to display format (e.g., "vitaminB12" -> "Vitamin B12")
                const displayTestName = testName.replace(/([A-Z])/g, ' $1') // Add space before capitals
                                            .replace(/^./, str => str.toUpperCase()) // Capitalize first letter
                                            .replace(/(\d+)/g, ' $1'); // Add space before numbers
                
                const itemElement = document.createElement('div');
                itemElement.className = 'signif_item';
                itemElement.innerHTML = `
                    <span class="bullet_point"></span>
                    <span class="bold_txt">${{displayTestName}}:</span> ${{testResult}}
                `;
                labTestsContainer.appendChild(itemElement);
            }}

           
        }};
        // Toggle collapsible sections
        document.addEventListener('DOMContentLoaded', function() {{
        document.querySelectorAll('.banner').forEach(banner =>{{
            const expandIcon = banner.querySelector('.expand-icon');
            if (!expandIcon) return;
            
            const content = banner.parentElement.querySelector('.collapsible-content');
            
            // Initialize all as collapsed
            content.classList.remove('expanded');
            expandIcon.textContent = '▼';
            expandIcon.classList.remove('expanded');
            
            banner.addEventListener('click', function() {{
                const isExpanded = content.classList.toggle('expanded');
                expandIcon.classList.toggle('expanded');
                expandIcon.textContent = isExpanded ? '▲' : '▼';
            }});
        }});

        // Populate data
        populateData();

        }}
        
    </script>
</body>
</html>

    """