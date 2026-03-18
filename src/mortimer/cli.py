"""Click CLI for the Mortimer RAG assistant."""
import json

import click

from mortimer.pipeline.rag import RAGPipeline

_DEFAULT_ARXIV_URLS = [
    "https://arxiv.org/pdf/1706.03762",
    "https://arxiv.org/pdf/2005.14165",
    "https://arxiv.org/pdf/2303.08774",
]


@click.group()
def cli() -> None:
    """Mortimer: RAG document assistant for arxiv PDFs."""


@cli.command()
@click.argument("urls", nargs=-1)
def ingest(urls: tuple[str, ...]) -> None:
    """Download and index PDFs from URLS (defaults to 3 arxiv papers)."""
    url_list = list(urls) if urls else _DEFAULT_ARXIV_URLS
    pipeline = RAGPipeline()
    results = pipeline.ingest(url_list)

    for result in results:
        if result.total_chunks == 0:
            click.echo(f"Skipped (already indexed): {result.title}")
        else:
            click.echo(f"Indexed '{result.title}': {result.total_chunks} chunks")


@cli.command()
@click.argument("question")
def ask(question: str) -> None:
    """Ask QUESTION against the indexed documents. Prints JSON response."""
    pipeline = RAGPipeline()
    response = pipeline.query(question)
    click.echo(json.dumps(response.model_dump(), indent=2))


@cli.command()
def reset() -> None:
    """Clear all documents from the vector store."""
    pipeline = RAGPipeline()
    pipeline.reset()
    click.echo("Vector store cleared.")
