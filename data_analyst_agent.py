import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from openai import OpenAI
import io
import textwrap
import contextlib
import os
import re

# Streamlit UI setup
st.set_page_config(page_title="🧠 Data Analyst Agent", layout="wide")
st.title("📊 Data Analyst Agent with Structured Analysis")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])

# Load DataFrame
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("🧾 Dataset Preview")
    st.dataframe(df.head())

    # st.subheader("📐 Schema Information")
    # schema_info = df.dtypes.astype(str) + ", Unique: " + df.nunique().astype(str) + ", Nulls: " + df.isnull().sum().astype(str)
    # st.dataframe(schema_info.rename("Info"))

    # Create schema summary string for GPT
    schema_str = "\n".join([f"- {col}: {dtype}, {df[col].nunique()} unique, {df[col].isnull().sum()} nulls"
                            for col, dtype in df.dtypes.items()])

    # Add example queries
    st.subheader("💡 Example Queries")
    examples = {
        "List top 10 best-selling items this month": "Filter by current month, group by items, sum quantities, sort descending, select top 10, and visualize with a bar chart",
        "Show me the trend of total sales over time": "Group by date, calculate daily sales, and plot a time series",
        "Identify items with low inventory": "Find items where current quantity is below a threshold and display them",
        "Compare performance across different regions": "Group by region, calculate metrics, and create a comparative visualization"
    }
    
    example_buttons = st.columns(len(examples))
    for i, (example, description) in enumerate(examples.items()):
        with example_buttons[i]:
            if st.button(f"Example {i+1}", help=description):
                query = example
                st.session_state.query = example
    
    user_api_key = st.text_input(
        "🔑 Enter your OpenAI API Key",
        type="password",
        key="api_key_input",
        placeholder="sk-xxxxxxxxxxxx..."
    )
    
    # Input for analysis question
    query = st.text_input("💬 Ask your data analysis question", key="query", value=st.session_state.get("query", ""))

    if query:
        if not user_api_key.strip():
            st.error("⚠️ Please provide your OpenAI API Key first.")
        else:
            with st.spinner("🤖 Processing your analysis (this may take a moment)..."):
                # Initialize OpenAI client with user's API key
                client = OpenAI(api_key=user_api_key.strip())
                # Prompt for GPT with schema and structured planning focus
                prompt = f"""
    You are an expert data analyst who creates clear, step-by-step analysis plans and writes clean, error-free pandas and matplotlib code.

    The user has uploaded a DataFrame named `df`. Here is the schema:

    {schema_str}

    The user asked:
    \"{query}\"

    First, create a numbered step-by-step plan (like example below) that clearly outlines the analytical process. Each step should be concise but descriptive.

    Example plan structure:
    1. Filter the data to include only transactions from the current month.
    2. Group the filtered data by 'itemName' and calculate the total sales volume for each item.
    3. Sort the items by total sales volume in descending order.
    4. Select the top 10 items based on total sales volume.
    5. Create a bar chart to visualize the top 10 best-selling items.

    After creating the plan, generate valid Python code that implements each step. Structure your code with clear step comments:

    ```python
    # Step 1: Filter data for current month
    current_month = datetime.now().month
    df_current_month = df[df['date'].dt.month == current_month]
    print("Current month data:", df_current_month.shape)

    # Step 2: Group by item and calculate sales volume
    # ... and so on
    ```

    IMPORTANT PLOTTING REQUIREMENTS:
    - DO NOT include `plt.show()` in your code (Streamlit handles visualization display)
    - Always end plots with `plt.close()` to free resources
    - For bar plots, always convert any numeric index or x values to strings using .astype(str)
    - When plotting with item names or IDs, ensure they are strings with: df['column_name'] = df['column_name'].astype(str)
    - For cleaner plots, consider limiting label length: df['short_name'] = df['long_name'].str[:20] + '...'
    - Always add appropriate titles and labels to plots
    - Use plt.tight_layout() for better spacing

    Don't include code to read the CSV file. The DataFrame `df` is already loaded.
    Make sure each step in your code has a print statement showing the relevant output.
    For visualizations, include proper titles, labels, and formatting.
    """

                # Call OpenAI API
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": """You are an expert data analyst who creates clear, step-by-step analysis plans and writes clean, error-free pandas and matplotlib code."
                        - Do NOT include any matplotlib backend configuration (e.g., matplotlib.use(...))
                        - Always end plots with plt.close() to free resources"""},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )

                gpt_output = response.choices[0].message.content

                # Split into plan and code
                if "```python" in gpt_output:
                    plan = gpt_output.split("```python")[0].strip()
                    code = gpt_output.split("```python")[1].split("```")[0].strip()
                else:
                    plan = gpt_output
                    code = ""

                # Extract just the numbered steps for the plan display
                plan_steps = re.findall(r'^\d+\..*', plan, re.MULTILINE)
                
                # Display the plan with nicer formatting
                st.subheader("🧠 Analysis Plan")
                for step in plan_steps:
                    st.markdown(f"- {step}")

                st.subheader("💻 Generated Python Code")
                st.code(code, language="python")

                # Clean code (avoid reading files)
                cleaned_code_lines = []
                for line in code.split("\n"):
                    if "pd.read_csv" in line:
                        continue
                    cleaned_code_lines.append(line)
                cleaned_code = "\n".join(cleaned_code_lines)

                # Preprocess the code to fix common plotting issues

                def preprocess_code(code):
                    # Remove plt.show() calls
                    code = re.sub(r'plt\.show\(\)', '# plt.show() removed by Streamlit handler', code)
                    # Ensure plt.close() is called at the end of plotting steps
                    code = re.sub(r'plt\.close\(\)', '# Already handled by Streamlit', code, flags=re.IGNORECASE)
                    return code

                cleaned_code = preprocess_code(cleaned_code)

                # Function to split code into steps
                def process_with_steps(code):
                    steps = []
                    current_step = []
                    
                    for line in code.strip().split('\n'):
                        if line.strip().startswith('# Step') and current_step:
                            steps.append('\n'.join(current_step))
                            current_step = [line]
                        else:
                            current_step.append(line)
                    
                    if current_step:
                        steps.append('\n'.join(current_step))
                    
                    # If no clear steps found, treat the whole code as one step
                    if len(steps) <= 1:
                        # Try to split by comments
                        steps = []
                        current_step = []
                        for line in code.strip().split('\n'):
                            if line.strip().startswith('#') and line.strip() != '#' and current_step:
                                steps.append('\n'.join(current_step))
                                current_step = [line]
                            else:
                                current_step.append(line)
                        
                        if current_step:
                            steps.append('\n'.join(current_step))
                    
                    # If still no clear steps, use the whole code
                    if len(steps) <= 1:
                        steps = [code]
                    
                    return steps

                # Process code into steps
                code_steps = process_with_steps(cleaned_code)

                # Execute the code steps one by one
                st.subheader("🛠️ Step-by-Step Execution")
                env = {
                    "df": df.copy(),
                    "pd": pd,
                    "plt": plt,
                    "datetime": datetime,
                    "np": np,
                    "st": st,
                    "re": re
                }
                final_results = {}
                            
                # Create expanders for each step
                for i, step_code in enumerate(code_steps):
                    step_num = i + 1
                    
                    # Extract step description from comments
                    step_description = f"Step {step_num}"
                    # Extract detailed description from comments
                    comment_match = re.search(r'# Step \d+:?\s*(.*)', step_code)
                    if comment_match:
                        step_description = comment_match.group(1).strip()
                    else:
                        # Fallback to first comment line
                        comment_match = re.search(r'#\s*(.*)', step_code)
                        if comment_match:
                            step_description = comment_match.group(1).strip()
                    
                    with st.expander(f"{step_num}. {step_description}", expanded=True):
                        st.code(step_code, language="python")
                        
                        output = io.StringIO()
                        
                        try:
                            has_plot = 'plt.' in step_code
                            
                            # Create safe execution wrapper for plotting code
                            if has_plot:
                                indented_code = textwrap.indent(step_code, ' ' * 8)
                                indented_retry = textwrap.indent(step_code, ' ' * 12)
                                plot_safe_code = (
                                    "def safe_execute():\n"
                                    "    try:\n"
                                    f"{indented_code}\n"
                                    "    except (TypeError, ValueError) as e:\n"
                                    "        if 'must be an instance of str' in str(e):\n"
                                    "            print('Auto-converting columns to strings...')\n"
                                    "            for df_name in [v for v in locals() if isinstance(locals()[v], pd.DataFrame)]:\n"
                                    "                df_to_convert = locals()[df_name]\n"
                                    "                df_to_convert = df_to_convert.astype({col: str for col in df_to_convert.select_dtypes(include=['object']).columns})\n"
                                    "                locals()[df_name] = df_to_convert\n"
                                    f"{indented_retry}\n"
                                    "        else:\n"
                                    "            raise\n"
                                    "    except Exception as e:\n"
                                    "        print(f'Error: {type(e).__name__} - {str(e)}')\n"
                                    "        raise\n"
                                    "safe_execute()\n"
                                )
                                exec_code = plot_safe_code
                            else:
                                exec_code = step_code
                            
                            # Execute code and capture output
                            with contextlib.redirect_stdout(output):
                                exec(exec_code, env)
                                
                            # Handle plotting
                            # In the step execution loop:
                            if has_plot:
                                fig = plt.gcf()
                                if fig.get_axes():
                                    # Add title if missing
                                    if not fig.texts and not fig.axes[0].get_title():
                                        plt.title(f"Step {step_num}: {step_description}")
                                    # Use Streamlit's native plotting
                                    st.pyplot(fig)
                                    plt.close(fig)  # Explicitly close the figure
                            
                            # Display stdout output
                            stdout = output.getvalue().strip()
                            if stdout:
                                st.text("Execution Output:")
                                st.text(stdout)
                            
                            # Display last variable assignment
                            last_assign = re.findall(r'(\w+)\s*=', step_code)
                            if last_assign:
                                var_name = last_assign[-1]
                                if var_name in env:
                                    st.subheader(f"Result: {var_name}")
                                    if isinstance(env[var_name], pd.DataFrame):
                                        st.dataframe(env[var_name].head(10))
                                        st.download_button(
                                            label="Download",
                                            data=env[var_name].to_csv(index=False),
                                            file_name=f"step_{step_num}_{var_name}.csv"
                                        )
                                    else:
                                        st.write(env[var_name])
                        
                        except Exception as e:
                            st.error(f"⚠️ Error in Step {step_num}:")
                            st.exception(e)
                            # Attempt basic recovery for common issues
                            if "must be an instance of str" in str(e) and i < len(code_steps)-1:
                                st.warning("Attempting automatic string conversion fix...")
                                # Add string conversion helper to environment
                                env["auto_str"] = lambda df: df.astype(str)
                            else:
                                st.stop()  # Halt execution on critical errors
                # Display final summary if we have results
                if final_results:
                    st.subheader("📊 Analysis Summary")
                    
                    if 'data' in final_results and isinstance(final_results['data'], pd.DataFrame):
                        st.write("Final dataset:")
                        st.dataframe(final_results['data'])
                        
                        # Add download button for final results
                        csv = final_results['data'].to_csv(index=False)
                        st.download_button(
                            label="Download final results",
                            data=csv,
                            file_name="analysis_results.csv",
                            mime="text/csv",
                        )
                    
                    # Show conclusion based on the analysis
                    st.subheader("🔍 Key Insights")
                    insight_prompt = f"""
                        Based on the analysis of the user question: "{query}"
                        and the data analysis performed, provide 3-5 key insights or conclusions in bullet point format.
                        Be concise and focus on what the analysis reveals.
                    """
                    
                    insight_response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are an expert data analyst who can extract key insights from analysis results."},
                            {"role": "user", "content": insight_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=250
                    )
                    
                    insights = insight_response.choices[0].message.content
                    st.markdown(insights)

else:
    st.info("📥 Please upload a CSV file to begin your analysis.")
    
    # Show some sample data structure info
    st.subheader("💡 This agent can help you:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Analyze trends** in your data
        - **Find patterns** and correlations
        - **Create visualizations** of key metrics
        - **Identify outliers** or anomalies
        """)
        
    with col2:
        st.markdown("""
        - **Generate summaries** of large datasets
        - **Compare groups** within your data
        - **Calculate statistics** and metrics
        - **Forecast future trends** based on historical data
        """)