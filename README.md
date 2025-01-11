#Odia Text BPE Tokenizer

This is a simple implementation of a BPE tokenizer for Odia text. It uses the `OdiaBPETokenizer` class to encode and decode text. The tokenizer is trained on a dataset of Odia text and saves the result to `odia_tokenizer.json`file.

## Usage

The training process is done in the `odiatokenizer.py` file.
To start the training process, you can run the `main` function in the `odiatokenizer.py` file. This will read the dataset, train the tokenizer, and save the resulting tokenizer to a file.

The `app.py` file is a simple Gradio interface that allows you to encode and decode text using the tokenizer.

A debug_tokenizer.py file is also provided test the functionalities in local system.

Total number of tokens in the tokenizer is 5000.Out of which 80 are odia characters, 10 are special characters and 4220 are subwords.
The compression ratio on tasted data is well above 3.2

The tokenizer is trained on a dataset 10m characters of odia text.

The link to the huggingface model is https://huggingface.co/spaces/amlanr66/OdiaBPETokenizer

Sample input text on Huggingface space showing compression ratio


![HuggingFaceApp](https://github.com/user-attachments/assets/7da320fa-502f-45a7-9503-c83da25fde76)
