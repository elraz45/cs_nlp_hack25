import requests
from bs4 import BeautifulSoup
import os
from openai import OpenAI
import dotenv

dotenv.load_dotenv(dotenv_path="connection.env")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/"

#MODEL = "google/gemma-3-27b-it:free"
MODEL = "meta-llama/llama-3.3-70b-instruct:free"
SMALL_MODEL = "meta-llama/llama-3.2-1b-instruct:free"
MODEL_STRUCTURED_OUTPUT = "google/gemini-flash-1.5-8b"


client = OpenAI(
                base_url=OPENROUTER_API_URL,
                api_key=OPENROUTER_API_KEY,
            )


def ask_model(prompt, model=MODEL, max_tokens=500):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def extract_text_from_webpage(url):
    """
    Extract text content from a web page using BeautifulSoup.
    Args:
        url (str): The URL of the web page to extract text from
    Returns:
        str: The extracted text content from the web page
    """
    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove script and style elements from the parsed HTML tree
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text content
    text = soup.get_text()

    # Clean up the text by removing extra whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    return text


def summarize_text(text_to_summarize: str, model: str = MODEL) -> str:
    """
    Summarize text using OpenRouter API.
    Args:
        text_to_summarize (str): The text to summarize
        model (str): model url to use
    Returns:
        str: The summarized text
    """
    prompt = f"""
    Please summarize the following text into a list of short sentences about specific events or people.
    The length of the list should be between 3 and 5 topics.
    Write only the list of topics, no other text.
    Write without any formatting, just the list of topics.
    Text:\n {text_to_summarize}\n
    Summary:"""
    response = ask_model(prompt, model)
    return response.strip()


def summarize_text_structured_output(text_to_summarize: str, model: str = MODEL) -> str:
    prompt = f"""
        Please summarize the following text into a list of short sentences about specific events or people.
        The length of the list should be between 3 and 5 topics. 
        Write only the list of structured objects in the specified json format, no other text.
        The main entity field should contain the name of the main person in the news item.
        The news item field should be a standalone sentence describing the news item.
        Text:\n {text_to_summarize}\n"""

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "topics",
            "strict": True,
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "main_entity": {
                            "type": "string",
                            "description": "Main person in news item"
                        },
                        "news_sentence": {
                            "type": "string",
                            "description": "A sentence describing the news item about the topic"
                        }
                    },
                    "required": ["main_entity", "news_sentence"],
                    "additionalProperties": False
                },
                "minItems": 3,
                "maxItems": 5
            }
        }
    }

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt, 'provider': {'require_parameters': 'true'}}],
        response_format=response_format,
        max_tokens=500
    )
    return response.choices[0].message.content


def generate_fake_news(text: str, model: str = MODEL):
    prompt = f"""
    Please generate a fake news article based on the following text.
    The article should contradict the information in the text.
    The article should be 50 words long.
    The article subject should be the first topic in the text.
    {text}
    """
    return ask_model(prompt, model=model)


if __name__ == '__main__':
    url = "https://edition.cnn.com/entertainment"
    #url2 = 'https://www.rollingstone.com/music/music-news/'
    text = extract_text_from_webpage(url)
    
    print("=== SUMMARY ===")
    print(summarize_text(text, MODEL))
    
    print("\n=== STRUCTURED SUMMARY ===")
    print(summarize_text_structured_output(text, MODEL_STRUCTURED_OUTPUT))

    print("=== FAKE NEWS ===")
    print(generate_fake_news(text, MODEL))
