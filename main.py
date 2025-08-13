import uuid
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from workflow import multi_agent_final_graph

async def main():
    """
    Main conversation loop with interrupt-and-resume functionality.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    interrupted = False

    print("Welcome to Customer Support!")
    print("Type 'exit' to quit the conversation.\n")

    while True:
        try:
            if not interrupted:
                user_input = input("You: ")
                if user_input.lower() == 'exit':
                    print("Thank you for using Customer Support. Goodbye!")
                    break
                input_package = {"messages": [HumanMessage(content=user_input)]}
            else:
                user_input = input("You (provide info to continue): ")
                if user_input.lower() == 'exit':
                    print("Thank you for using Customer Support. Goodbye!")
                    break
                input_package = {"messages": [HumanMessage(content=user_input)]}

            stream_events = multi_agent_final_graph.astream_events(
                input_package, config, version="v1"
            )

            async for event in stream_events:
                event_type = event.get("event", "")

                if event_type == "on_chain_end":
                    output = event.get("data", {}).get("output", {})
                    messages = output.get("messages", [])
                    if messages and isinstance(messages[-1], AIMessage):
                        print(f"\nAssistant: {messages[-1].content}\n")
                    interrupted = False

                elif event_type == "on_chain_stream" and event.get("name") == "human_input":
                    chunk = event.get("data", {}).get("chunk", {})
                    messages = chunk.get("messages", [])
                    if messages and isinstance(messages[-1], AIMessage):
                        print(f"\nAssistant: {messages[-1].content}\n")
                    interrupted = True
                    break

        except (ValueError, KeyError) as e:
            print(f"\nAn application error occurred: {e}")
            print("Please try again or type 'exit' to quit.\n")
        except KeyboardInterrupt:
            print("\n\nConversation interrupted. Type 'exit' to quit.")
        except openai.APIError as e:
            print(f"\nOpenAI API error: {e}")
        except psycopg2.Error as e:
            print(f"\nDatabase error: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Please try again or type 'exit' to quit.\n")


if __name__ == "__main__":
    asyncio.run(main())
