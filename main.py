import os
import time
from dotenv import load_dotenv
from agents.research_agent import ResearchAgent
from agents.chart_agent import ChartAgent

# Load environment variables
load_dotenv()

def check_api_keys():
    """Check if required API keys are set in environment variables."""
    missing_keys = []
    
    if not os.getenv("GROQ_API_KEY"):
        missing_keys.append("GROQ_API_KEY")
    
    if not os.getenv("TAVILY_API_KEY"):
        missing_keys.append("TAVILY_API_KEY")
    
    if missing_keys:
        print("Error: The following API keys are missing from your .env file:")
        for key in missing_keys:
            print(f"- {key}")
        print("\nPlease add these keys to your .env file and try again.")
        return False
    
    return True

def main():
    """Main function to run the collaborative AI agent system."""
    print("=" * 50)
    print("AI Data Analysis Collaboration Agents")
    print("=" * 50)
    
    # Check API keys
    if not check_api_keys():
        return
    
    # Initialize agents
    try:
        print("\nInitializing Research Agent...")
        research_agent = ResearchAgent()
        
        print("Initializing Chart Agent...")
        chart_agent = ChartAgent()
        
        print("\nAgents initialized successfully!")
    except Exception as e:
        print(f"\nError initializing agents: {str(e)}")
        return
    
    # Main interaction loop
    while True:
        print("\n" + "=" * 50)
        query = input("\nEnter your query (or 'exit' to quit): ")
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("\nExiting the application. Goodbye!")
            break
        
        if not query.strip():
            print("Please enter a valid query.")
            continue
        
        try:
            # Step 1: Research Agent collects data
            print("\n[1/4] Research Agent is collecting data...")
            search_results = research_agent.search(query)
            print(f"     Found {len(search_results)} relevant sources.")
            
            # Step 2: Research Agent summarizes data
            print("\n[2/4] Research Agent is summarizing the data...")
            data_summary = research_agent.summarize_for_chart(query, search_results)
            print("     Data summarization complete.")
            
            # Step 3: Chart Agent generates code
            print("\n[3/4] Chart Agent is generating visualization code...")
            chart_code = chart_agent.generate_chart_code(query, data_summary)
            print("     Code generation complete.")
            
            # Step 4: Chart Agent executes code
            print("\n[4/4] Chart Agent is executing the code and generating the chart...")
            result = chart_agent.execute_chart_code(chart_code, data_summary)
            
            # Display results
            if result["error"]:
                print(f"\nError executing chart code: {result['error']['message']}")
                print("\nError details:")
                print(result["error"]["traceback"])
            else:
                if result["chart_path"]:
                    print(f"\nChart generated successfully and saved to: {result['chart_path']}")
                else:
                    print("\nCode executed successfully, but no chart was generated.")
                    
            # Display data summary
            print("\nData Summary:")
            print("-" * 50)
            print(data_summary)
            print("-" * 50)
            
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
    
if __name__ == "__main__":
    main() 