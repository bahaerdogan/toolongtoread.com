from transformers import BartTokenizer, BartForConditionalGeneration
import torch

class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

    def summarize_text(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=1024,
            truncation=True,
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, max_length=500, early_stopping=True)

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary