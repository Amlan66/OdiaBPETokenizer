import json
from odiatokenizer import OdiaBPETokenizer

def load_tokenizer():
    with open('odia_tokeniser.json', 'r', encoding='utf-8') as f:
        id_to_token = json.load(f)
    #print("Loaded token_to_id:", token_to_id)  # Debugging line
    return OdiaBPETokenizer.from_id_to_token(id_to_token)

def main():
    # Load the tokenizer
    tokenizer = load_tokenizer()

    # Sample text to encode and decode
    sample_text = "ଓଡ଼ିଶା ଲୋକଙ୍କର ଦୁର୍ଦ୍ଦଶା, ସ୍ଵତନ୍ତ୍ର ଓଡ଼ିଶା ପ୍ରଦେଶ ଗଠନ, ଅହିଂସା ଆନ୍ଦୋଳନ, ମହାତ୍ମା ଗାନ୍ଧୀଙ୍କ ବାର୍ତ୍ତା ଓ ଜାତୀୟ କଂଗ୍ରେସର ଆଭିମୁଖ୍ୟ ପ୍ରଚାର ତଥା ଜନସାଧାରଣଙ୍କ"

    # Encode the sample text
    encoded_tokens = tokenizer.encode(sample_text)
    print("Encoded Tokens:", encoded_tokens)

    # Decode the tokens back to text
    decoded_text = tokenizer.decode(encoded_tokens)
    print("Decoded Text:", decoded_text)

    # Calculate compression ratio
    input_length = len(sample_text)
    encoded_length = len(encoded_tokens)
    compression_ratio = input_length / encoded_length if encoded_length > 0 else 0
    print("Compression Ratio:", compression_ratio)

if __name__ == "__main__":
    main() 