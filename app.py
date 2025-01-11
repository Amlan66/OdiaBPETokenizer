import gradio as gr
import json
from odiatokenizer import OdiaBPETokenizer

# Load the tokenizer from the JSON file
def load_tokenizer():
    with open('odia_tokeniser.json', 'r', encoding='utf-8') as f:
        id_to_token = json.load(f)
    return OdiaBPETokenizer.from_id_to_token(id_to_token)

tokenizer = load_tokenizer()

def encode_text(input_text):
    # Encode the input text
    encoded_tokens = tokenizer.encode(input_text)
    
    # Create a mapping of input text parts to token IDs
    token_mapping = {}
    index = 0
    for token_id in encoded_tokens:
        subword = tokenizer.id_to_token.get(token_id, '<UNK>')
        token_mapping[subword] = token_id
        index += len(subword)

    # Create a color-coded HTML string for the encoded text
    color_coded_text = ''.join(
        f'<span style="color:#{hash(subword) & 0xFFFFFF:06x}">{subword}</span>'
        for subword in token_mapping.keys()
    )

    # Decode the tokens back to text
    decoded_text = tokenizer.decode(encoded_tokens)
    
    # Calculate compression ratio
    input_length = len(input_text)
    encoded_length = len(encoded_tokens)
    compression_ratio = input_length / encoded_length if encoded_length > 0 else 0
    
    return token_mapping, compression_ratio, decoded_text, color_coded_text

# Set up Gradio interface
iface = gr.Interface(
    fn=encode_text,
    inputs=gr.Textbox(label="Input Odia Text", placeholder="Type Odia text here..."),
    outputs=[
        gr.JSON(label="Encoded Tokens"),
        gr.Number(label="Compression Ratio"),
        gr.Textbox(label="Decoded Text", placeholder="Decoded text will appear here...", interactive=False),
        gr.HTML(label="Color-Coded Encoded Text")
    ],
    title="Odia Text Encoder",
    description="Enter Odia text to get the encoded tokens, compression ratio, and decoded text."
)

if __name__ == "__main__":
    iface.launch() 