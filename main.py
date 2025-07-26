"""
Main entry point for the CoderLLM application.
Manages command-line interface and stateful conversation loop.
"""

import uuid
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from workflow import multi_agent_final_graph


async def main():
    """
    Main conversation loop with interrupt-and-resume functionality.
    """
    # Generating session ID for the entire conversation
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Tracking interrupt state
    is_interrupted = False
    
    print("Welcome to Customer Support!")
    print("Type 'exit' to quit the conversation.\n")
    
    while True:
        try:
            # Get user input based on current state
            if not is_interrupted:
                # Normal input prompt
                user_input = input("You: ")
                if user_input.lower() == 'exit':
                    print("Thank you. Shutting Down!")
                    break
                
                # Package input as message dictionary
                input_package = {"messages": [HumanMessage(content=user_input)]}
            else:
                # Interrupted state - different prompt
                user_input = input("You (provide info to continue): ")
                if user_input.lower() == 'exit':
                    print("Thank you for using CoderLLM. Goodbye!")
                    break
                
                # For interrupt resumption, send raw string
                input_package = user_input
            
            # Process through the graph using astream_events
            stream_events = multi_agent_final_graph.astream_events(input_package, config, version="v1")
            async for event in stream_events:
                event_type = event.get("event", "")
                
                # Handle chain completion
                if event_type == "on_chain_end":
                    # Extract and display final AI message
                    output = event.get("data", {}).get("output", {})
                    messages = output.get("messages", [])
                    
                    if messages and isinstance(messages[-1], AIMessage):
                        print(f"\nAssistant: {messages[-1].content}\n")
                    
                    # Reset interrupt state
                    is_interrupted = False
                
                # Handle interruption
                elif event_type == "on_chain_stream" and event.get("name") == "human_input":
                    # Extract the interrupt message
                    chunk = event.get("data", {}).get("chunk", {})
                    messages = chunk.get("messages", [])
                    
                    if messages and isinstance(messages[-1], AIMessage):
                        print(f"\nAssistant: {messages[-1].content}\n")
                    
                    # Set interrupted state and break inner loop
                    is_interrupted = True
                    break
                    
        except KeyboardInterrupt:
            print("\n\nConversation interrupted. Type 'exit' to quit.")
            continue
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again or type 'exit' to quit.\n")
            continue


if __name__ == "__main__":
    asyncio.run(main())
