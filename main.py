from file_reader import read_file
from text_summarizer import TextSummarizer

if __name__ == "__main__":
    summarizer = TextSummarizer()
    text = read_file(r"D:\\MyWork\\toolongtoread.com\\filetomark.txt")
    if text:
        summary = summarizer.summarize_text(text)
        print("Summary: ", summary)