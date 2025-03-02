import sys
import os
import traceback
from io import StringIO
import contextlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class PythonREPL:
    """A tool for executing Python code in a REPL-like environment."""
    
    def __init__(self, charts_dir="charts"):
        """Initialize the Python REPL tool.
        
        Args:
            charts_dir (str): Directory where charts will be saved
        """
        self.charts_dir = charts_dir
        # Create charts directory if it doesn't exist
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
    
    @contextlib.contextmanager
    def capture_output(self):
        """Capture stdout and stderr."""
        stdout, stderr = StringIO(), StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = stdout, stderr
            yield stdout, stderr
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
    
    def execute(self, code, filename=None):
        """Execute the given Python code and save any generated charts.
        
        Args:
            code (str): Python code to execute
            filename (str, optional): Filename for saving the chart. If None, a default name will be used.
            
        Returns:
            dict: A dictionary containing execution results and any errors
        """
        # Clear any existing plots
        plt.close('all')
        
        # Prepare result dictionary
        result = {
            "output": "",
            "error": None,
            "chart_path": None
        }
        
        # Execute the code
        with self.capture_output() as (stdout, stderr):
            try:
                exec(code, globals())
                result["output"] = stdout.getvalue()
                
                # Check if a plot was created
                if plt.get_fignums():
                    # Generate filename if not provided
                    if filename is None:
                        filename = f"chart_{len(os.listdir(self.charts_dir)) + 1}.png"
                    
                    # Ensure filename has .png extension
                    if not filename.endswith('.png'):
                        filename += '.png'
                    
                    # Save the figure
                    chart_path = os.path.join(self.charts_dir, filename)
                    plt.savefig(chart_path, bbox_inches='tight')
                    result["chart_path"] = chart_path
                    plt.close()
            except Exception as e:
                result["error"] = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
        
        return result 