# Multi-Agent Customer Support System

This project implements a sophisticated, multi-agent customer support system using LangGraph and LangChain. It is designed to handle a variety of customer queries by routing them to specialized agents, ensuring efficient and accurate responses.

## Features

- **Multi-Agent Architecture**: Deploys specialized agents for different domains (e.g., music, invoices) supervised by a master agent.
- **Stateful Conversations**: Maintains conversation state, allowing for context-aware interactions and long-term memory.
- **Dynamic Routing**: The supervisor agent intelligently routes user queries to the appropriate sub-agent based on the conversation history.
- **Extensible**: Easily add new agents to handle additional domains.

## Getting Started

### Prerequisites

- Python 3.11+
- Poetry for dependency management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/onurunaall/multi-agent-system.git](https://github.com/onurunaall/multi-agent-system.git)
    cd multi-agent-system
    ```

2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```

3.  **Set up your environment variables:**
    Create a `.env` file by copying the example file:
    ```bash
    cp .env.example .env
    ```
    Then, add your OpenAI API key to the `.env` file.

### Usage

To start the customer support chat interface, run the following command:

```bash
poetry run python main.py