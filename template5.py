import gradio as gr
import openai
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Tuple
from datetime import datetime

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY
client = OpenAI(api_key=API_KEY)

# Initialize conversation log
conversation_log = []


# Function to handle input and generate responses from two models
def respond_from_two_models(
        message: str,
        system_message: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
):
    # Prepare the conversation history for both models
    messages = [{"role": "system", "content": system_message}]
    messages.append({"role": "user", "content": message})

    # Response from the first model
    response_model_1 = ""
    stream_1 = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:uw-sbel::A6Rd900h",
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    )
    for chunk in stream_1:
        if chunk.choices[0].delta.content is not None:
            response_model_1 += chunk.choices[0].delta.content

    # Response from the second model
    response_model_2 = ""
    stream_2 = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:uw-sbel::A6Rd900h",  # Example for another model
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    )
    for chunk in stream_2:
        if chunk.choices[0].delta.content is not None:
            response_model_2 += chunk.choices[0].delta.content

    # Log the conversation
    conversation_log.append({
        "user_message": message,
        "model_1_response": response_model_1,
        "model_2_response": response_model_2
    })

    # Automatically save the conversation
    save_conversation()

    return response_model_1, response_model_2


# Function to capture feedback
def feedback_good(feedback_text):
    feedback = {
        "rating": "GOOD",
        "feedback_text": feedback_text
    }
    conversation_log.append({"feedback": feedback})
    save_conversation()
    return f"Feedback: The answer was marked as GOOD. Additional feedback: {feedback_text}"


# Function to reset the textbox after submission
def reset_textbox():
    return ""


# Function to enable the chatbot interface after user consent
def enable_inputs():
    return gr.update(visible=False), gr.update(visible=True)


# Function to save conversation log as a JSON file automatically
def save_conversation():
    # Use current timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_log_{timestamp}.json"

    # Save conversation log to JSON
    with open(filename, "w") as f:
        json.dump(conversation_log, f, indent=4)

    print(f"Conversation automatically saved as {filename}")


# Main Gradio interface setup
with gr.Blocks() as demo:
    # User consent container
    with gr.Column(elem_id="user_consent_container") as user_consent_block:
        accept_checkbox = gr.Checkbox(visible=False)
        js = "(x) => confirm('By clicking \"OK\", I agree that my data may be published or shared.')"

        # Accordion for user consent
        with gr.Accordion("User Consent for Data Collection, Use, and Sharing", open=True):
            gr.HTML("""
            <div>
                <p>By using this chatbot, which is powered by OpenAI's API, you acknowledge and agree to the following terms regarding data collection and use:</p>
                <ol>
                    <li><strong>Data Collection:</strong> We may collect the inputs you provide, the generated outputs, and certain technical details about your device (e.g., browser type, OS, IP address).</li>
                    <li><strong>Data Use:</strong> Collected data may be used for research, improving services, and security purposes.</li>
                    <li><strong>Data Sharing:</strong> Your data may be shared with third parties for analysis and reporting.</li>
                    <li><strong>Data Retention:</strong> Data may be retained as necessary.</li>
                </ol>
                <p>By continuing to use the chatbot, you provide explicit consent for data collection and use as described above.</p>
            </div>
            """)
            accept_button = gr.Button("I Agree")

    # Main chatbot interface (hidden until consent is given)
    with gr.Column(elem_id="col_container", visible=False) as main_block:
        with gr.Row():
            user_input = gr.Textbox(label="Enter your prompt here", placeholder="Type your input and press Submit",
                                    lines=2)

        # Additional settings
        system_message = gr.Textbox(value="You are a friendly Chatbot.", label="System message", lines=2)
        max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
        temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")

        # Output for both models
        with gr.Row():
            model_1_output = gr.Textbox(label="Model 1 Response")
            model_2_output = gr.Textbox(label="Model 2 Response")

        # Submit button to trigger both models
        submit_button = gr.Button("Submit Prompt")

        # Run the models when submit button is clicked
        submit_button.click(respond_from_two_models,
                            inputs=[user_input, system_message, max_tokens, temperature, top_p],
                            outputs=[model_1_output, model_2_output])

        # Feedback section
        feedback_text = gr.Textbox(placeholder="Additional feedback (optional)")
        good_feedback_button = gr.Button("Good 👍")
        not_good_feedback_button = gr.Button("Not Good 👎")
        feedback_output = gr.Textbox(label="Feedback Output")

        # Handle feedback submission with additional feedback
        good_feedback_button.click(feedback_good, inputs=[feedback_text], outputs=feedback_output)

    # Click event for the consent button
    accept_button.click(None, None, accept_checkbox, js=js, queue=False)
    accept_checkbox.change(fn=enable_inputs, inputs=[], outputs=[user_consent_block, main_block], queue=False)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(share=True)
