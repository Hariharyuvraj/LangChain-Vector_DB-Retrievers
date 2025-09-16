# Imports
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import WikipediaRetriever

# Initialize retriever (Wikipedia)
wiki_retriever = WikipediaRetriever(lang="en", top_k_results=2)

# Query Wikipedia
query = "Indian IPL cricket team"
docs = wiki_retriever.invoke(query)   # use .invoke(), not .get_relevant_documents()

# Your OpenAI API Key
OPENAI_API_KEY = "Paste your openai api key"

# Store docs in Chroma
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

vectorstore = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Convert vectorstore into retriever
retriever = vectorstore.as_retriever()

# Query the retriever
results = retriever.get_relevant_documents("Who is the captain of all IPL cricket teams?")

# Print results in multiline
for i, r in enumerate(results, 1):
    print(f"\n🔹 Result {i}:\n{r.page_content[:300]}...\n")
