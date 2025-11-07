from openai import OpenAI

SDOH_CLINICAL_PROMPT  = """
                You are tasked with reviewing a patient’s clinical note along with their Social Determinants of Health (SDoH) items. Your goal is to generate a comprehensive, concise, and human-understandable clinical report.
                For your report, follow these steps:

                ### Step 1: Review the SDoH items
                Carefully analyze the patient’s SDoH information, focusing on the following items that are strongly linked to cognitive impairments:

                1. **Social Isolation**
                2. **Nutrition / Food Insecurity**
                3. **Physical Activity Level**
                4. **Education Level**
                5. **Smoking / Tobacco Use**
                6. **Alcohol Use**

                If the values in the SDoH items match the **threatening values** listed below, make sure to **explicitly mention them in the report**:

                - **Social Isolation**: Severely isolated
                - **Nutrition / Food Insecurity**: Often skips meals due to lack of food
                - **Physical Activity Level**: Sedentary
                - **Education Level**: No formal education, Primary
                - **Smoking / Tobacco Use**: Current
                - **Alcohol Use**: Alcoholic

                ### Step 2: Analyze the Clinical Note
                Review the patient’s clinical note and identify any points that indicate an **imminent threat** or signs that could be directly and strongly associated with cognitive impairments.

                ### Step 3: Generate the Clinical Report
                Your clinical report should contain the following:

                - **SDoH items**: Identify and list only the items that match the **threatening values** listed above. **Highlight and bold these** values in the report in a clear and readable format as **Bold** structure.

                - **Imminent threats from the clinical note**: Identify and summarize **any clinical note points** that strongly suggest risks for cognitive impairments.

                - **Negative Points**: For any **negative conditions and Imminent threats** you found , be sure to format them as **Bold**. This ensures they stand out and become bolded in the report.

                ### Step 4: Formatting and Tone
                Maintain a clinical, factual tone and  be **natural, clear, and concise** —avoid bullet points or itemized lists, I want a coherent and cohesive paragraph.
                It is very important that you **DO NOT** explicitly mention in the report paragraph that the patient has or does not have cognitive impairments.
                It is very important to include ALL information but **BOLD**:
                  1. SDoH items that match the **threatening values** listed above. (e.g The patient is ** Severely isolated **)
                  2. **Negative** or **Imminent threats** you found in the clinical report. (e.g The paitient chief complaint is ** headache **)
                Eliminate redundant terms and keep the narrative concise, do not add any explanation, note, title, etc. only a paragraph as the report.
                Use the example below to guide tone, structure, and style.

                ### Example output:
                The patient reports having a **Sedentary** lifestyle, primarily due to mobility limitations, and feels **socially isolated**, often relying on neighbors for occasional transportation assistance.

                ---
                ### SDOH items:
                {sdoh_dict}
                ---
                ### Clinical Notes:
                {clinical_notes}
                ---
                ### Output Report:
"""

LAB_TESTS_PROMPT = """
        You are tasked with reviewing a patient's lab test results, and determine those items that are strongly associated with cognitive impairments and have values that exceed the **danger zone** or **imminent threat** thresholds.

        Here are the lab test items that are strongly associated with cognitive impairments and their associated dangerous thresholds:

        1. **Interleukin-6 (IL-6)**: >3 pg/mL
        2. **Monocyte Chemoattractant Protein-1 (MCP-1)**: >500 pg/mL
        3. **Fibrinogen**: >400 mg/dL
        4. **Aβ42**: <450 pg/mL
        5. **NfL**: >10 pg/mL
        6. **Tau**: >300 pg/mL
        7. **Aβ42/Aβ40**: <0.05
        8. **Apolipoprotein B (ApoB)**: >90 mg/dL
        9. **Homocysteine**: >15 µmol/L
        10. **Vitamin B12**: <200 pg/mL
        11. **Systolic Blood Pressure (SBP)**: >140 mmHg
        12. **Diastolic Blood Pressure (DBP)**: >90 mmHg
        13. **Low-Density Lipoprotein (LDL)**: >160 mg/dL
        14. **Fasting Glucose**: >100 mg/dL
        15. **Ankle-Brachial Index (ABI)**: <0.9
        16. **Albuminuria**: >30 mg/g creatinine

        ### Your Task:
        1. Review the provided lab test results carefully.
        2. For any lab test item that is present in the list above and has a value that exceeds or is below the dangerous threshold, choose that lab test item and its value.
        3. Ensure you include both the **lab test item** and its **value with units** exactly as given, and match it carefully to the threshold provided.
        4. If a test item’s value does not exceed the threshold, ignore it.
        5. Return the result as a list of items in a format like [{{lab test items: lab test value}}].
        6. Avoid adding any extra title, explanation, comments, etc..

        ### Example Output:
        [{{"Fasting Glucose": "130 mg/dL"}}, {{"Tau": "350 pg/mL"}}, {{"Vitamin B12": "180 pg/mL"}}]

        ---
        ### Lab Tests items:
        {lab_test_dict}
        ---
        ### Output:
"""
CLINICAL_FUNCIONAL_PROMPT = """
    You are tasked with reviewing a patient's Clinical and Functional Overview, and determine those items that are strongly associated with cognitive impairments and have values that exceed the **danger zone** or **imminent threat** thresholds, along with their appropriate subcategory.

    Here are the items that are strongly associated with cognitive impairments categorized into subcategories and their associated dangerous thresholds:

    ### 1. Demographic and Basic Information
    - **Body Mass Index (BMI)**: < 18.5 or ≥ 30

    ### 2. Psychological and Behavioral
    - **Depression Severity (PHQ-9 Category)**: Severe (20–27)
    - **Mental Conditions Diagnosed**: PTSD or Anxiety

    ### 3. Functional Status
    - **Activities of Daily Living (ADL) Category**: Dependent
    - **Self-Care Ability**: Unable to perform self-care

    ### 4. Cognitive Symptoms
    - **Understanding Verbal Information (gg0100a)**:  Unable to understand verbal information
    - **Memory Problems (gg0100c)**: Severe
    - **Decision-Making Ability (gg0100d)**: Severely impaired

    ### 5. Physiological
    - **Diabetes Diagnosis**: Diabetes with complications
    - **Myocardial Infarction (Heart Attack)**: Yes
    - **Congestive Heart Failure**: Yes
    - **Cerebrovascular Disease (e.g., Stroke)**: Yes
    - **Chronic Lung Disease**: Severe COPD/Restrictive disease
    - **Renal Disease**: End-stage renal disease
    - **Recent Falls or High Fall Risk**: Yes
    - **Recent Weight Loss**: Severe (>10%)
    - **General Frailty Indicator**: Yes
    - **Sleep Disturbances**: Yes
    - **Dyspnea Severity (Shortness of Breath)**: At rest

    ### 6. Medical Interventions and Therapies
    - **Currently Receiving Chemotherapy**: Yes
    - **Currently Receiving Dialysis**: Yes

     ### Your Task:
        1. Review the provided lab test results carefully.
        2. For any lab test item that is present in the list above and has a value that exceeds or is below the dangerous threshold, choose that lab test item and its value.
        3. Ensure you include both the **lab test item** and its **value with units** exactly as given, and match it carefully to the threshold provided.
        4. If a test item’s value does not exceed the threshold, ignore it.
        5. Return the result as a list of items in a format like [{{lab test items: lab test value}}].


    ### Your Task:
    1. Review the patient's Clinical and Functional Overview.
    2. For any item that is present in the list above and have threatening value from the list, choose that item, its value, and its **subcategory**.
    3. Ensure you include both the **item**, its **value**, and its **subcategory** exactly as given, and match it carefully to the threshold provided.
    4. If a test item’s value does not exceed the threshold, ignore it.
    5. Return the result as a list of items in a format like [{{item: [value , category]}}].
    6. Avoid adding any extra title, explanation, comments, etc..

    **Example Output**:
    [{{"Currently Receiving Chemotherapy": ["130 mg/dL","Medical Interventions and Therapies"]}}]

    ---
    ### Clinical and Functional Overview items:
    {clinical_functional_dict}
    ---
    ### Output:

"""
SIG_FACTORS_PROMPT =  """
  Given the following four types of interpretation, your task is to extract the most significant factor from each interpretation that is strongly and directly associated
  with cognitive impairments. Compare all the items in each interpretation and select the one that stands out the most based on its direct relationship to cognitive decline or impairment.
  Your response should focus on selecting the item from each category that has the most notable connection to cognitive impairment symptoms.

  Interpretations:

  - SDoH (Social Determinants of Health):
  {SDoH}

  - Clinical (Medical History and Symptoms):
  {Clinical}

  - Acoustic (Speech Features and Patterns):
  {Acoustic}

  - Linguistic (Language Use and Structure):
  {Linguistic}

  Instructions for the Model:
  1. For each interpretation type (SDoH, Clinical, Acoustic, Linguistic), compare the individual items presented and evaluate which one most strongly relates to cognitive impairments.
  2. The selected item should be the one that most directly reflects or correlates with cognitive decline or dementia-related symptoms.
  3. Write the selected item as a natural, human-undesrstandable, and concise **COMPLETE SENTENCE**.
  4. Output the selected significant factors for each interpretation type in the following format:
  [
      "Acoustic_selected",
      "Linguistic_selected",
      "SDoH_selected",
      "Clinical_selected",
  ]
  5. Do not add any extra comment, explanation, title, etc. ONLY include the selected items in the said format.
  """



def generate_clinical_functional(clinical_functional_dict,openai_config):

    prompt = CLINICAL_FUNCIONAL_PROMPT.format(clinical_functional_dict =clinical_functional_dict )

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

def generate_lab_test(lab_test_items,openai_config):

    prompt = LAB_TESTS_PROMPT.format(lab_test_dict =lab_test_items )

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

def generate_SDoH_text(SDoH,clinical_notes,openai_config):

    prompt = SDOH_CLINICAL_PROMPT.format(sdoh_dict =SDoH,clinical_notes=clinical_notes )

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


def generate_significant_factors(SDoH,clinical_functional,linguistic,acoustic,openai_config):

    prompt = SIG_FACTORS_PROMPT.format(SDoH=SDoH,Clinical=clinical_functional,Linguistic=linguistic,Acoustic=acoustic)

    model = OpenAI(
            api_key=openai_config['api_key'],
            base_url=openai_config.get('base_url')
        )

    def _generate_openai(prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        return model.chat.completions.create(
            model=kwargs.get('model', "meta-llama/Llama-3.1-405B"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 2048)
        )

    return _generate_openai(prompt).choices[0].message.content

