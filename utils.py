"""
Utility functions for the CoderLLM application.
"""

from langchain_core.runnables.graph import MermaidDrawMethod


def save_graph_diagram(graph, output_filename="graph.png"):
    """
    Save a visual representation of the graph to a file.
    
    Args:
        graph: The LangGraph graph object
        output_filename: The filename to save the diagram to
    """
    try:
        # Get the graph visualization as PNG bytes
        png_data = graph.get_graph().draw_mermaid_png()
        
        # Write the bytes to file
        with open(output_filename, "wb") as f:
            f.write(png_data)
        
        print(f"Graph diagram saved to {output_filename}")
        
    except Exception as e:
        print(f"Could not save graph diagram using primary method: {e}")
        
        try:
            # Fallback method using pyppeteer
            import nest_asyncio
            nest_asyncio.apply()
            
            # Use the synchronous method with pyppeteer draw method
            png_data = graph.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.PYPPETEER
            )
            
            with open(output_filename, "wb") as f:
                f.write(png_data)
            
            print(f"Graph diagram saved to {output_filename} using fallback method")
            
        except Exception as fallback_error:
            print(f"Could not save graph diagram: {fallback_error}")


def get_langgraph_docs_retriever():
    """
    Create a retriever for LangGraph documentation.
    Note: This function is not used in this command-line application.
    """
    # This function would implement document retrieval functionality
    # but is not needed for the current CLI implementation
    pass
