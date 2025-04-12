import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model_name = "text-tmbedding-3-large")
Settings.chunk_size = 512
Settings.chunk_overlap = 64


load_dotenv(override=True)
def main():
    print("Hello from uber-10k-chatbot!")


if __name__ == "__main__":
    main()
