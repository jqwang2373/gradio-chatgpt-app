import gradio as gr
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Tuple

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY
client = OpenAI(api_key=API_KEY)


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

    # Yield the response tokens as they come
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            response += token
            yield response


# Function to capture feedback
def capture_feedback(is_good_answer, feedback_text):
    if is_good_answer:
        return f"Feedback: The answer was marked as good. Additional feedback: {feedback_text}"
    else:
        return f"Feedback: The answer was marked as not good. Additional feedback: {feedback_text}"


# Function to reset the textbox after submission
def reset_textbox():
    return ""


# Function to enable the chatbot interface after user consent
def enable_inputs():
    return gr.update(visible=False), gr.update(visible=True)


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
        chatbot = gr.ChatInterface(
            respond,
            additional_inputs=[
                gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
                gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
                gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
                gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
            ],
        )

        # Feedback section
        with gr.Column(visible=True):
            feedback_good_answer = gr.Checkbox(label="Was this answer good?")
            feedback_text = gr.Textbox(placeholder="Additional feedback (optional)")
            feedback_button = gr.Button("Submit Feedback")
            feedback_output = gr.Textbox(label="Feedback Output")

        # Handle feedback submission
        feedback_button.click(capture_feedback, [feedback_good_answer, feedback_text], feedback_output)

    # Click event for the consent button
    accept_button.click(None, None, accept_checkbox, js=js, queue=False)
    accept_checkbox.change(fn=enable_inputs, inputs=[], outputs=[user_consent_block, main_block], queue=False)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(share=True)
