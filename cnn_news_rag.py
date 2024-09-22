import requests
from bs4 import BeautifulSoup
from newspaper import Article
from gtts import gTTS
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import nltk

# Download necessary NLTK resources
nltk.download('punkt')

# Step 1: Fetch CNN Homepage and Extract Links
def get_cnn_homepage():
    url = "https://www.cnn.com"
    response = requests.get(url)
    return response.text

def extract_article_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    article_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/') and '/2024/' in href:  # Filter for recent articles
            full_url = f"https://www.cnn.com{href}"
            if full_url not in article_links:
                article_links.append(full_url)
    return article_links[:5]  # Limit to 5 articles

# Step 2: Extract Article Details
def extract_article_details(url):
    article = Article(url)
    article.download()
    article.parse()
    return {
        'title': article.title,
        'text': article.text
    }

# Step 3: Text to Speech Conversion
def text_to_speech(text, filename):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

# Step 4: RaG Implementation for Article Retrieval and Summarization
def index_articles(articles):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    texts = [article['text'] for article in articles]
    embeddings = model.encode(texts, convert_to_tensor=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, texts

def retrieve_articles(query, index, texts):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = model.encode([query], convert_to_tensor=False)
    D, I = index.search(np.array(query_embedding), 5)
    retrieved_articles = [texts[i] for i in I[0]]
    return retrieved_articles

def generate_summary(retrieved_articles):
    openai.api_key = 'your_openai_api_key'  # Replace with your OpenAI API key
    prompt = "Summarize the following news articles:\n"
    for article in retrieved_articles:
        prompt += f"\nArticle: {article}\n"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    return response['choices'][0]['text']

# Main Function to Run the Program
def main():
    print("Fetching CNN homepage...")
    homepage_html = get_cnn_homepage()
    print("Extracting article links...")
    article_links = extract_article_links(homepage_html)

    print(f"Found {len(article_links)} articles. Extracting details...")
    articles = []
    for link in article_links:
        article_details = extract_article_details(link)
        articles.append(article_details)
        print(f"Extracted: {article_details['title']}")

    print("Indexing articles...")
    index, texts = index_articles(articles)

    query = input("Enter a topic or keyword for news retrieval: ")
    retrieved_articles = retrieve_articles(query, index, texts)
    print(f"Found {len(retrieved_articles)} relevant articles.")

    print("Generating summary...")
    summary = generate_summary(retrieved_articles)
    print("Summary:")
    print(summary)

    # Convert summary to audio
    filename = "news_summary.mp3"
    text_to_speech(summary, filename)
    print(f"Summary converted to audio: {filename}")

if __name__ == "__main__":
    main()
