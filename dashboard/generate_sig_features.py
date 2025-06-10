import pandas as pd
import base64

def generate_html_sif_features(sig_features_path,pieChart="data/pie_chart.png"):

    df = pd.read_excel(sig_features_path)

    sig_features = {
        row["Feature"]: {"value": row["Value"], "color": row["Color"]}
        for _, row in df.iterrows()
    }
    # Function to encode file to base64
    def encode_file_base64(path):
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    # Encode files
    piechart_b64 = encode_file_base64(pieChart)

    # Convert to list of dictionaries and sort in descending order
    sorted_features = sorted(
        [{"name": name, "value": data["value"], "color": data["color"]} 
        for name, data in sig_features.items()],
        key=lambda x: x["value"],
        reverse=True
    )

    # Generate HTML for each bar row
    bar_rows = []
    for feature in sorted_features:
        row_html = f"""
        <div class="bar-row">
            <div class="feature-name">{feature['name']}</div>
            <div class="bar-container">
                <div class="bar" 
                    style="width: {feature['value']}%; 
                            background-color: {feature['color']};">
                </div>
            </div>
            <div class="value-label">{feature['value']}</div>
        </div>
        """
        bar_rows.append(row_html)



    return f"""
    
    <!DOCTYPE html>
    <html>
        <head>
            <style>
                body {{
                    background-color: #E5F1F3;
                    font-size: 1em;
                    font-family: Arial, Helvetica, sans-serif;
                    margin: 0;
                    padding: 0;
                    text-align: center;
                    color:#1E3658;
                }}
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                .title{{
                    color: #1E3658;
                    padding: 8rem 0 2rem 0;

                }}
                .chart-container {{
                    width: 80%;
                    margin: 20px auto;
                    font-family: Arial, sans-serif;
                }}
                .bar-row {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 10px;
                }}
                .feature-name {{
                    width: 150px;
                    text-align: left;
                    padding-right: 10px;
                    /* font-weight: bold; */
                }}
                .bar-container {{
                    flex-grow: 1;
                    height: 30px;
                    background-color: #f0f0f0;
                    border-radius: 3px;
                    overflow: hidden;
                }}
                .bar {{
                    height: 100%;
                    transition: width 0.5s ease;
                    display: flex;
                    align-items: center;
                    padding-left: 10px;
                    box-sizing: border-box;
                    color: white;
                    font-weight: bold;
                    text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
                }}
                .value-label {{
                    width: 50px;
                    padding-left: 10px;
                    text-align: right;
                }}
                #chart{{
                    border-bottom: 2px solid #1E3658;
                }}
                .heading{{
                    display: flex;
                    flex-direction: row;
                    align-content: center;
                    align-items: center;
                    justify-content: center;
                    gap: 10rem;
                    }}
            </style>
        </head>
        <body>
            <div class="heading">
                <h1 class="title">20 Most Important Factors in Cognitive Impairment Diagnosis</h1>
                <img src="data:image/png;base64,{piechart_b64}" alt="Pie chart of the modality contribution">
            </div>
            
            <div class="chart-container" id="chart"> {"".join(bar_rows)}</div>

            <script>
            </script>
        </body>
    </html>
    """