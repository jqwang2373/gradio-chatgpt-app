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


# Function to handle user input and generate response from GPT model
def respond(
        message: str,
        history: List[Tuple[str, str]],
        system_message: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
):
    # Prepare messages with system message and conversation history
    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})

    response = ""

    # Stream the response from OpenAI API
    stream = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:uw-sbel::A6Rd900h",
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    )

    # Build the response and log the conversation
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            response += token

    # Log the user message and bot response
    conversation_log.append({"user_message": message, "bot_response": response})

    # Automatically save the conversation after each response
    save_conversation()

    return response


# Function to capture feedback when "Good" is clicked
def feedback_good(feedback_text):
    feedback = {
        "rating": "GOOD",
        "feedback_text": feedback_text
    }
    conversation_log.append({"feedback": feedback})
    save_conversation()
    return f"Feedback: The answer was marked as GOOD. Additional feedback: {feedback_text}"


# Function to capture feedback when "Not Good" is clicked
def feedback_not_good(feedback_text):
    feedback = {
        "rating": "NOT GOOD",
        "feedback_text": feedback_text
    }
    conversation_log.append({"feedback": feedback})
    save_conversation()
    return f"Feedback: The answer was marked as NOT GOOD. Additional feedback: {feedback_text}"


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
                <p>By using this chatbot(ChronoGPT-mini), which is developed by University of Wisconsin-Madison Simulation-Based Engineering Lab(UW-SBEL) AI team, you acknowledge and agree to the following terms regarding data collection and use:</p>
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
        chatbot = gr.ChatInterface(
            respond,
            additional_inputs=[
                gr.Textbox(value="You are a PyChrono expert.", label="System message"),
                gr.Slider(minimum=1, maximum=4096, value=2048, step=1, label="Max new tokens"),
                gr.Slider(minimum=0.1, maximum=1.0, value=0.1, step=0.1, label="Temperature"),
                gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Top-p (nucleus sampling)"),
            ],
            examples=[
                ["Run a PyChrono simulation of a sedan driving on a flat surface with a detailed vehicle dynamics model."],
                ["Run a real-time simulation of an HMMWV vehicle on a bumpy and textured road."],
                ["Set up a Curiosity rover driving simulation on flat, rigid ground in PyChrono."],
                ["Simulate a FEDA vehicle driving on rigid terrain in PyChrono."],
            ],
            cache_examples=False,
        )

        # Feedback section with two buttons and additional feedback textbox
        with gr.Column(visible=True):
            good_feedback_button = gr.Button("Good 👍")
            not_good_feedback_button = gr.Button("Not Good 👎")
            feedback_text = gr.Textbox(placeholder="Additional feedback (optional)")
            feedback_output = gr.Textbox(label="Feedback Output")

        # Handle feedback submission with additional feedback
        good_feedback_button.click(feedback_good, inputs=[feedback_text], outputs=feedback_output)
        not_good_feedback_button.click(feedback_not_good, inputs=[feedback_text], outputs=feedback_output)

    # Click event for the consent button
    accept_button.click(None, None, accept_checkbox, js=js, queue=False)
    accept_checkbox.change(fn=enable_inputs, inputs=[], outputs=[user_consent_block, main_block], queue=False)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(share=True)
