from transformers import pipeline
from dataCleaning import data_cleaning

file_path = "test1.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

text = data_cleaning(text)

summarizer = pipeline('summarization')
summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
print(summary[0]['summary_text'])