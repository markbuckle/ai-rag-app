import argparse

# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

# {context} is the pieces of information we received from the database
# {query} is the actual query itself
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # load the chroma DB using the path that we used in create_database and use the embedding function.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # once the db is loaded (above) you can search for the chunk in the DB that best matches the query and specify the # (k) of results (best matches) we want to retrieve
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # Return type of search. Each Tuple contains a document and its relevance score
    # List[Tuple[Document, float]]

    # If there are no matches or the relevance score is below a threshold, we can return early. This will help us make sure we find good/relevant info first before moving into the next step of the process
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    # code to actual use this data to create the prompt by formatting the template with our keys. After running this you should have a single (long) string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # LLM of your choice for your response
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    # provide references back to your source data
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
