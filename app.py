import gradio as gr
import json
from odia_tokenizer import OdiaBPETokenizer

# Load the tokenizer from the JSON file
def load_tokenizer():
    with open('odia_tokenizer.json', 'r', encoding='utf-8') as f:
        token_to_id = json.load(f)
    return OdiaBPETokenizer(token_to_id)

tokenizer = load_tokenizer()

def encode_text(input_text):
    # Encode the input text
    encoded_tokens = tokenizer.encode(input_text)
    
    # Calculate compression ratio
    input_length = len(input_text)
    encoded_length = len(encoded_tokens)
    compression_ratio = input_length / encoded_length if encoded_length > 0 else 0
    
    return encoded_tokens, compression_ratio

# Set up Gradio interface
iface = gr.Interface(
    fn=encode_text,
    inputs=gr.Textbox(label="Input Odia Text", placeholder="Type Odia text here..."),
    outputs=[
        gr.JSON(label="Encoded Tokens"),
        gr.Number(label="Compression Ratio")
    ],
    title="Odia Text Encoder",
    description="Enter Odia text to get the encoded tokens and compression ratio."
)

if __name__ == "__main__":
    iface.launch() 