import openai

def main():
    # Configure OpenAI client for local server
    client = openai.OpenAI(
        base_url="http://localhost:8080/v1",  # Server URL
        api_key="sk-no-key-required"  # No API key required
    )

    print("Chatbot started! Type 'exit' to end the program.")
    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Add user input to the conversation history
        conversation_history.append({"role": "user", "content": user_input})

        # Send request to the local LLaMA server
        response = client.chat.completions.create(
            model="LLaMA_CPP",  
            messages=conversation_history
        )

        # Extract response
        assistant_reply = response.choices[0].message.content
        print("Chatbot:", assistant_reply)

        # Save response to conversation history
        conversation_history.append({"role": "assistant", "content": assistant_reply})

if __name__ == "__main__":
    main()
