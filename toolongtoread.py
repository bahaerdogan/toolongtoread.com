from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Read the file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Summarize the text
def summarize_text(text, model, tokenizer):
    # Tokenize the text
    inputs = tokenizer.encode_plus(
        text,
        max_length=1024,  # You can increase this value to handle larger texts
        truncation=True,
        return_tensors='pt',  # Return as PyTorch tensors
    )
    
    # Generate a summary of the text using the model
    with torch.no_grad():
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=500, early_stopping=True)
    
    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    print("Summary: ", summary)

# Main function
def main():
    # File path
    file_path = r"D:\\MyWork\\toolongtoread.com\\filetomark.txt"
    
    # Load the model and tokenizer
    model_name = "facebook/bart-large-cnn"  # Choose the model name you want to use
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    # Read the file
    text = read_file(file_path)
    
    # Summarize the text
    summarize_text(text, model, tokenizer)

if __name__ == "__main__":
    main()