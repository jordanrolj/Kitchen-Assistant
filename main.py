import os
from dotenv import load_dotenv
from agent import ConversationalAgent  # Import your agent from its file

# Load environment variables (e.g., API keys)
load_dotenv()

def main():
    # Initialize the conversational agent
    agent = ConversationalAgent()

    # Example messages to test the different tools and interactions
    test_messages = [
        "Suggest a recipe",  # Should invoke RAG tool
        "What is the nutritional value of the previous recipe",  # Should invoke the nutrition tool
    ]

    # Loop through each test message and print the agent's response
    for message in test_messages:
        print(f"User: {message}")
        response = agent.handle_input(message)
        print(f"Agent: {response}\n")

if __name__ == "__main__":
    main()