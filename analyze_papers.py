from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import requests
import arxiv
from io import BytesIO
from pypdf import PdfReader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


class MyState(TypedDict):

    topic: str
    alex_papers: list
    arxiv_papers: list
    platform_available: str
    relevant_papers: list
    paper_texts: list
    analysis: str
    next_action: str
    research_gaps: str
    paper_analysis: dict


def user_input(state: MyState):

    user_input = input('Enter Topic: ')

    return {
        "topic": user_input,
        "alex_papers": [],
        "arxiv_papers": [],
        "paper_texts": [],
        "paper_analysis": []
    }

def search_open_alex(state: MyState):

    query = state['topic']
    url = "https://api.openalex.org/works"

    params = {

        "search": query,
        "per-page": 10

    }

    response = requests.get(url, params=params)
    print("Status Code:", response.status_code)

    data = response.json()

    for paper in data["results"]:

        print("\nTitle:", paper["title"])
        print("Year:", paper["publication_year"])
        print("Citations:", paper["cited_by_count"])
        print("DOI:", paper["doi"])

    papers = data['results']

    return {'alex_papers': papers}


def search_arxiv(state: MyState):

    query = state["topic"]
    search = arxiv.Search(query=query,max_results=10)

    client = arxiv.Client()
    papers = []

    for paper in client.results(search):
        print("\nTitle:", paper.title)
        print("PDF:", paper.pdf_url)
        papers.append(paper)

    print("\nNumber of papers found:", len(papers))

    return {'arxiv_papers': papers}


def read_papers(state: MyState):

    arxiv_papers = state["arxiv_papers"]
    alex_papers = state["alex_papers"]

    paper_texts = list(state.get("paper_texts", []))


    # Read arXiv papers

    for paper in arxiv_papers:

        print("\nReading:", paper.title)
        print("Downloading:", paper.pdf_url)

        try:

            response = requests.get(
                paper.pdf_url,
                timeout=30
            )

            response.raise_for_status()

            pdf_file = BytesIO(response.content)

            reader = PdfReader(pdf_file)

            text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text


            paper_texts.append({
                "title": paper.title,
                "text": text
            })

            print("Successfully read:", paper.title)


        except requests.exceptions.RequestException as e:

            print("Could not download:", paper.title)
            print("Error:", e)


        except Exception as e:

            print("Could not read:", paper.title)
            print("Error:", e)


    # Read OpenAlex papers

    for paper in alex_papers:

        print("\nReading:", paper["title"])

        abstract = paper.get(
            "abstract_inverted_index"
        )

        if abstract:

            words = {}

            for word, positions in abstract.items():

                for position in positions:

                    words[position] = word

            text = " ".join(
                words[i]
                for i in sorted(words)
            )

        else:

            text = "(no abstract available)"


        paper_texts.append({
            "title": paper["title"],
            "text": text
        })


    print(
        "\nTotal papers read:",
        len(paper_texts)
    )


    return {
        "paper_texts": paper_texts
    }