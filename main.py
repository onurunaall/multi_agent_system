import uuid
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from workflow import multi_agent_final_graph

async def main():
    """Main conversation loop."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("Welcome to Customer Support!")
    print("Type 'exit' to quit the conversation.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Thank you for using Customer Support. Goodbye!")
                break

            input_package = {"messages": [HumanMessage(content=user_input)]}
            
            # Use ainvoke instead of astream_events for simpler handling
            result = await multi_agent_final_graph.ainvoke(input_package, config)
            
            # Extract the last AI message
            if result and "messages" in result:
                messages = result["messages"]
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, AIMessage):
                        print(f"\nAssistant: {last_message.content}\n")
                    else:
                        print(f"\nAssistant: {last_message.content}\n")
                else:
                    print("\nAssistant: I'm sorry, I didn't understand that.\n")
            else:
                print("\nAssistant: I'm sorry, something went wrong.\n")

        except KeyboardInterrupt:
            print("\n\nConversation interrupted. Type 'exit' to quit.")
        except ImportError as e:
            print(f"\nMissing dependency: {e}")
            print("Run 'poetry install' to install required packages.")
            break
        except ConnectionError as e:
            print(f"\nConnection error: {e}")
            print("Please check your internet connection and database status.")
        except Exception as e:
            print(f"\nSystem error: {e}")
            print("Please restart the application or contact support.")

if __name__ == "__main__":
    asyncio.run(main())