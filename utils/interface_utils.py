import pandas as pd
import matplotlib.pyplot as plt
import base64
import io

def generate_modality_pie_base64(contribution):
    contrib_dict = {
        "Acoustic": contribution[0],
        "Linguistic": contribution[1],
        "Demographic": contribution[2]
    }
    color_map = {
        "Linguistic": "#558755",
        "Acoustic": "#5883BE",
        "Demographic": "#eb9c1e"
    }

    labels = []
    sizes = []
    colors = []

    for key, value in contrib_dict.items():
        if value > 0:
            labels.append(key)
            sizes.append(float(value))
            colors.append(color_map.get(key, "#cccccc"))

    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.0f%%',
        startangle=140,
        textprops={'color': "#1E3658", 'fontsize': 20, 'weight': 'bold'},
        wedgeprops=dict(edgecolor='w')
    )

    # Set white color for inside percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(20)
        autotext.set_weight('bold')

    ax.axis('equal')  # Keep it circular

    # Save as base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def extract_patient_data(df):
    """
    Extracts structured sections from a two-column DataFrame. Handles missing 'Clinical Notes' gracefully.

    Args:
        df (pd.DataFrame): A dataframe with two columns: [Section, Value]

    Returns:
        dict: A dictionary with structured patient data categorized by section.
    """
    if len(df.columns) > 2:
        df = df.iloc[:, :2]

    # Rename columns
    df.columns = ['Section', 'Value']

    # Define section markers
    sections = {
        "Demographic and Basic Information": "Demographic and Basic Information",
        "Social Determinants of Health (SDoH)": "Social Determinants of Health (SDoH)",
        "Psychological and Behavioral": "Psychological and Behavioral",
        "Functional Status": "Functional Status",
        "Cognitive Symptoms": "Cognitive Symptoms",
        "Physiological": "Physiological",
        "Medical Interventions and Therapies": "Medical Interventions and Therapies",
        "Laboratory Tests & Biomarkers": "Laboratory Tests & Biomarkers",
        "Clinical Note": "Clinical Note"
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
            str(row['Section']).strip(): (
                '' if pd.isna(row['Value'])
                else str(row['Value']).strip()
            )
            for _, row in sub_df.iterrows()
            if str(row['Section']).strip().lower() != 'nan'}
    return structured_data




def read_excel_sample_data(excel_path,id=None):
    xls = pd.ExcelFile(excel_path, engine='openpyxl')
    if id:
        df = pd.read_excel(xls, sheet_name=id)
    else:
        df = pd.read_excel(xls)

    if len(df.columns) > 2:
        df = df.iloc[:, :2]

    new_df = pd.DataFrame({
    'Section': ['Demographic and Basic Information'] + df[df.columns[0]].tolist(),
    'Value': ['NaN'] + df[df.columns[1]].tolist()
    })

    return new_df

