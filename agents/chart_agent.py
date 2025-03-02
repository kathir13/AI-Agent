import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.python_repl import PythonREPL

# Load environment variables
load_dotenv()

class ChartAgent:
    """Agent for generating and executing Python code to create charts."""
    
    def __init__(self):
        """Initialize the Chart Agent with necessary components."""
        # Initialize Groq LLM for code generation
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=4000
        )
        
        # Initialize Python REPL for code execution
        self.python_repl = PythonREPL()
    
    def generate_chart_code(self, query, data_summary):
        """Generate Python code for creating a chart based on the data summary.
        
        Args:
            query (str): The original user query
            data_summary (str): Summarized data from the Research Agent
            
        Returns:
            str: Python code for generating the requested chart
        """
        # Create a prompt with clear instructions for the LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data visualization expert. Your task is to generate Python code 
            that creates charts based on provided data.
            
            Guidelines:
            1. Use matplotlib, pandas, and seaborn for data visualization
            2. Generate clean, well-commented code that is easy to understand
            3. Include code to properly format the chart (titles, labels, legends, etc.)
            4. Make the chart visually appealing with appropriate colors and styling
            5. Include code to handle the data in the format it's provided
            6. Ensure the code is complete and can run without additional input
            7. Focus on creating exactly the type of chart requested in the query
            8. Include error handling where appropriate
            9. Handle incomplete or missing data gracefully - filter out any missing values
            10. If data is incomplete, only use the available data points and mention this in the chart title
            
            IMPORTANT: Your output should be ONLY the raw Python code needed to generate the chart.
            DO NOT include any markdown formatting, code block delimiters (like ```python or ```), 
            or any other non-Python syntax. The code will be executed directly, so it must be valid Python code only."""),
            ("user", f"""Original Query: {query}
            
            Data Summary:
            {data_summary}
            
            Please generate Python code to create the chart requested in the original query
            using the data provided in the summary. The code should be complete and ready to execute.
            
            IMPORTANT: 
            1. The code should save the chart to a file but should NOT display it using plt.show()
            2. DO NOT include markdown formatting or code block delimiters (like ```python or ```)
            3. Return ONLY the raw Python code that can be executed directly
            4. Include proper error handling for data validation
            5. Handle incomplete or missing data gracefully - filter out any missing, null, or placeholder values
            6. If some data is missing, only use the available data and note this in the chart title
            """)
        ])
        
        try:
            # Generate code
            chain = prompt | self.llm | StrOutputParser()
            code = chain.invoke({})
            
            # Remove any markdown code block syntax if present
            code = self._clean_code(code)
            
            return code
        except Exception as e:
            print(f"Error generating code: {str(e)}. Using fallback prompt.")
            return self._generate_fallback_code(query, data_summary)
    
    def _generate_fallback_code(self, query, data_summary):
        """Generate fallback code when the main prompt fails.
        
        Args:
            query (str): The original user query
            data_summary (str): Summarized data from the Research Agent
            
        Returns:
            str: Simple Python code for generating a chart
        """
        # Create a simpler prompt without complex formatting
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a data visualization expert. Generate simple Python code to create a chart based on the data provided."),
            ("user", f"""Query: {query}
            
            Data: {data_summary}
            
            Generate Python code to create a chart. Use matplotlib, pandas, and seaborn.
            
            IMPORTANT:
            1. If creating a heatmap, use df.pivot_table(index='Row', columns='Column', values='Value') instead of df.pivot()
            2. The code should save the chart to a file but should NOT display it using plt.show()
            3. Return ONLY the raw Python code that can be executed directly
            """)
        ])
        
        try:
            # Generate code
            chain = prompt | self.llm | StrOutputParser()
            code = chain.invoke({})
            
            # Remove any markdown code block syntax if present
            code = self._clean_code(code)
            
            return code
        except Exception as e:
            # If even the fallback fails, return a very basic template
            print(f"Fallback also failed: {str(e)}. Using basic template.")
            return """
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

# Ensure charts directory exists
if not os.path.exists('charts'):
    os.makedirs('charts')

# Parse the data from the summary
# This assumes the data is in a table format
lines = data_summary.split('\n')
table_lines = [line for line in lines if '|' in line]

# Extract header
headers = table_lines[0].split('|')
headers = [h.strip() for h in headers if h.strip()]

# Extract data
data = []
for line in table_lines[2:]:  # Skip header and separator
    if '|' in line:
        row = line.split('|')
        row = [cell.strip() for cell in row if cell.strip()]
        if len(row) >= 3:  # Ensure we have enough columns
            data.append(row)

# Create DataFrame
if data and len(headers) >= 3:
    # Assuming format: City | Month | Temperature
    df = pd.DataFrame({
        headers[0]: [row[0] for row in data],
        headers[1]: [row[1] for row in data],
        headers[2]: [float(row[2].replace('°F', '').strip()) for row in data if row[2].replace('°F', '').strip().replace('.', '', 1).isdigit()]
    })
    
    # Create pivot table for heatmap
    pivot_data = df.pivot_table(index=headers[0], columns=headers[1], values=headers[2])
    
    # Set up the matplotlib figure
    plt.figure(figsize=(14, 8))
    
    # Generate the heatmap
    sns.heatmap(pivot_data, annot=True, fmt=".1f", linewidths=.5, cmap="YlOrRd")
    
    # Add title and labels
    plt.title('Heatmap of ' + headers[2] + ' by ' + headers[0] + ' and ' + headers[1], fontsize=16)
    plt.xlabel(headers[1], fontsize=12)
    plt.ylabel(headers[0], fontsize=12)
    
    # Save the figure
    plt.savefig('charts/heatmap.png', bbox_inches='tight')
    print("Heatmap saved to charts/heatmap.png")
else:
    print("Could not parse data for heatmap")
"""
    
    def _clean_code(self, code):
        """Clean the generated code by removing any markdown formatting.
        
        Args:
            code (str): The generated code
            
        Returns:
            str: Cleaned code without markdown formatting
        """
        # Remove markdown code block syntax if present
        if code.startswith("```python"):
            code = code.split("```python", 1)[1]
        if code.startswith("```"):
            code = code.split("```", 1)[1]
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]
            
        # Strip any leading/trailing whitespace
        code = code.strip()
        
        return code
    
    def execute_chart_code(self, code, data_summary, chart_name=None):
        """Execute the generated Python code to create and save the chart.
        
        Args:
            code (str): Python code to execute
            data_summary (str): Summarized data from the Research Agent
            chart_name (str, optional): Name for the chart file
            
        Returns:
            dict: Execution results including chart path and any errors
        """
        # Add basic error handling for common chart issues
        error_handling_code = f"""
# Ensure charts directory exists
import os
if not os.path.exists('charts'):
    os.makedirs('charts')

# Make data_summary available to the code
data_summary = '''{data_summary}'''

# Basic validation function
def validate_chart_data(data, labels=None):
    if labels is not None and len(data) != len(labels):
        raise ValueError(f"Data length ({{len(data)}}) must match labels length ({{len(labels)}})")
    return True

# Helper function to safely convert values to float
def safe_float_convert(value):
    if value is None or value == '-' or value == '':
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Remove percentage sign if present
        value = value.rstrip('%')
        try:
            return float(value)
        except ValueError:
            return None
    return None
"""
        
        # Combine error handling code with generated code
        enhanced_code = error_handling_code + "\n" + code
        
        # Execute the code
        result = self.python_repl.execute(enhanced_code, filename=chart_name)
        return result