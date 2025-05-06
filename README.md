/cognitive-impairment-explainability
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── /data/
│ ├── acoustic/ # Acoustic-specific data
│ │ ├── lxbg.json
│ │ ├── model_config.yaml
│ │ ├── pause_config.yaml
│ │ ├── qbce.json
│ │ ├── qnvo.json
│ │ ├── samples_df.csv
│ │ ├── vocal_features.csv
│ ├── linguistic/ # Linguistic-specific data
│ │ ├── model_config.yaml
│ │ ├── samples_df.csv
│
├── /acoustic_module/
│ ├── checkpoints/
│ ├── dataset/
│ │ ├── Dataset.py
│ ├── figs/
│ │ ├── lxbg_spectrogram.png
│ ├── generalMethods/
│ │ ├── acoustic_feature.py
│ ├── interpretation/
│ │ ├── interpretationReport.py
│ ├── model/
│ │ ├── Model.py
│ │ ├── ModelWrapper.py
│ │ ├── WeightsManager.py
│ ├── pauseExtraction/
│ │ ├── Pause_extraction.py
│ ├── test/
│ │ ├── test.py
│ ├── result.ipynb
│
├── /linguistic_module/
│ ├── dataset/
│ │ ├── Dataset.py
│ ├── figs/
│ │ ├── SHAP_qnvo.png
│ ├── generalMethods/
│ │ ├── modelDecisionAnalysis.py
│ ├── htmls/
│ │ ├── SHAP_lxbg.html
│ ├── Llama/
│ │ ├── Llama.py
│ ├── models/
│ │ ├── Model.py
│ ├── test/
│ │ ├── test_Shap.py
│ ├── trainer/
│ │ ├── Trainer.py
│ ├── results_SHAP.ipynb
│
├── /shared/
│ ├── SHAP/
│ │ ├── Shap.py
│ │ ├── text_visualization.py
│ ├── utils/
│ │ ├── Config.py
│ │ ├── dataset_utils.py
│ │ ├── Utils.py
│ ├── generalMethods/
│ │ ├── modelDecisionAnalysis.py # If common version, otherwise move to each module
│
├── /interactive_dashboard/
│ ├── dashboard.py # Streamlit/Dash/etc. app that combines both outputs
│ ├── components.py # Modular visual components (graphs, tables, etc.)
│ ├── utils.py
│ ├── /assets/ # HTML, CSS, or image files for UI
│
├── /docs/
│ ├── architecture.png
│ ├── usage_guide.md
│ ├── methodology.md
