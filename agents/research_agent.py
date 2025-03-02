import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

class ResearchAgent:
    """Agent for collecting and summarizing data from the web using Tavily search."""
    
    def __init__(self):
        """Initialize the Research Agent with necessary API clients."""
        # Initialize Tavily client for web search
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        
        # Initialize Groq LLM for summarization
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=4000
        )
    
    def search(self, query, max_results=5):
        """Perform a web search using Tavily.
        
        Args:
            query (str): The search query
            max_results (int): Maximum number of search results to return
            
        Returns:
            list: List of search results
        """
        search_results = self.tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results
        )
        return search_results.get("results", [])
    
    def summarize_for_chart(self, query, search_results):
        """Summarize search results in a format suitable for chart generation.
        
        Args:
            query (str): The original user query
            search_results (list): List of search results from Tavily
            
        Returns:
            str: Summarized data suitable for chart generation
        """
        # Extract content from search results
        content = ""
        for result in search_results:
            content += f"Source: {result.get('url', 'Unknown')}\n"
            content += f"Content: {result.get('content', '')}\n\n"
        
        # Create prompt for summarization
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data research assistant. Your task is to extract and organize 
            numerical data from search results to be used for chart generation.
            
            Guidelines:
            1. Focus on extracting precise numerical data relevant to the query
            2. Organize data in a clear, structured format (tables, lists, etc.)
            3. Include time periods, dates, or categories when relevant
            4. Ensure data is properly labeled
            5. If data is incomplete or inconsistent, note this clearly
            6. Format the data in a way that would be easy for a chart generation system to use
            7. Include only factual information from the sources, no opinions
            8. If the data appears in different formats or units across sources, standardize it
            
            Your output should be a well-structured summary of the numerical data found in the search results,
            ready to be used for generating the requested chart."""),
            ("user", f"""Original Query: {query}
            
            Search Results:
            {content}
            
            Please extract and organize the relevant numerical data from these search results
            that would be needed to generate the chart requested in the original query.
            Format the data in a clear, structured way that would be easy to use for chart generation.""")
        ])
        
        # Generate summary
        chain = prompt | self.llm | StrOutputParser()
        summary = chain.invoke({})
        
        return summary 