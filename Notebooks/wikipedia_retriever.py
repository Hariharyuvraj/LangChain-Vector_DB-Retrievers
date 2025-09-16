# Imports
from langchain_community.retrievers import WikipediaRetriever

# Initialize retriever
retriever = WikipediaRetriever(
    lang="en",          # language = English
    top_k_results=2     # how many results you want
)

# Example query
query = "Retrieval-Augmented Generation"

docs = retriever.invoke(query)

# if i use this code result will print in single line

print(docs)  # result print in single line only.