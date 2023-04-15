import re
import openai
from dotenv import load_dotenv
import os
import csv
from tqdm import tqdm 

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
    
with open('data/kjv.txt', 'r') as f:
    lines = f.readlines()

chapters = {} 

for line in lines:
    line = line.strip()

    reference, content = line.split(' ', 1)
    book_chapter, verse = reference.split(':')
    book, chapter = re.match(r"(\d?[a-zA-Z]+)([0-9]+)", book_chapter).groups()

    if book_chapter not in chapters:
        chapters[book_chapter] = {
            "book": book,
            "chapter": int(chapter),
            "verses": [int(verse)],
            "content": content, 
        }
    else:
        chapters[book_chapter]["verses"].append(int(verse))
        chapters[book_chapter]["content"] += f" {content}"

print(f"Creating embeddings for {len(chapters)} chapters")
for book_chapter, chapter in tqdm(chapters.items()):
    response = openai.Embedding.create(
        input=chapter["content"],
        model="text-embedding-ada-002",
    )
    chapters[book_chapter]["embedding"] = response['data'][0]['embedding']

with open('embeddings.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['book_chapter', 'book', 'chapter', 'verses', 'content', 'embedding']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for book_chapter, chapter in chapters.items():
        row = {
            'book_chapter': book_chapter,
            'book': chapter['book'],
            'chapter': chapter['chapter'],
            'verses': ','.join(str(v) for v in chapter['verses']),
            'content': chapter['content'],
            'embedding': chapter['embedding'],
        }
        writer.writerow(row)

# json format
# import json
# with open('embeddings.json', 'w') as outfile:
#     json.dump(chapters, outfile, ensure_ascii=False, indent=4)

