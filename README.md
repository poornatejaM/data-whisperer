# 🧠 Data Analyst Agent with GPT-4

An AI-powered data analysis tool that turns your questions into actionable insights. Upload your CSV data and get step-by-step analysis with Python code generation and visualizations.

---

## Features

✅ **Smart Analysis**  
- Automatically generates analysis plans and Python code using GPT-4  
- Handles data cleaning, transformations, and visualizations  
- Provides key business insights from the results  

✅ **Interactive Workflow**  
- Real-time code execution with error handling  
- Step-by-step breakdown of analysis processes  
- Downloadable datasets and final results  

✅ **Visualization Support**  
- Auto-formatted matplotlib plots  
- Automatic string conversion for plot compatibility  
- Built-in error recovery for common issues  

---

## Prerequisites

- Python 3.8+
- OpenAI API Key (get one [here](https://platform.openai.com/account/api-keys))
- Required Python packages (listed in `requirements.txt`)

---

## Installation

1. **Create Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
2. **Install Dependencies**  
```bash
pip install -r requirements.txt
```
## Run the streamlit app

```bash
streamlit run data_analyst_agent.py
```