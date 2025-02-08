from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def main():
    model_name = "microsoft/phi-2"  # Hugging Face model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print("Chatbot started! Type 'exit' to end the program.")
    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Add user input to the conversation history
        conversation_history.append(f"User: {user_input}")

        # Format the input with the conversation history
        prompt = "\n".join(conversation_history) + "\nAssistant:"

        # Generate a response from the model
        response = chat_pipeline(prompt, max_length=512, do_sample=True, temperature=0.7)

        # Extract the assistant's reply
        assistant_reply = response[0]["generated_text"].split("Assistant:")[-1].strip()
        print("Chatbot:", assistant_reply)

        # Save the assistant's reply
        conversation_history.append(f"Assistant: {assistant_reply}")

if __name__ == "__main__":
    main()
