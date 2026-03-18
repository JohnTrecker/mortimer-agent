"""Prompt template loading and message construction for RAG."""
from pathlib import Path

from mortimer.models.schemas import RetrievedChunk

_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"


def load_template(name: str) -> str:
    """Load a prompt template from the prompts/ directory.

    Args:
        name: Template name without extension (e.g. 'rag_system').

    Returns:
        Template string contents.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context string.

    Args:
        chunks: List of RetrievedChunk objects to format.

    Returns:
        Multi-line string with numbered, source-attributed chunks.
    """
    if not chunks:
        return ""

    parts = []
    for i, retrieved in enumerate(chunks, start=1):
        chunk = retrieved.chunk
        source_label = f"{chunk.metadata.source}"
        if chunk.metadata.section:
            source_label += f" | {chunk.metadata.section}"
        source_label += f" | page {chunk.metadata.page_number + 1}"
        parts.append(f"[{i}] ({source_label})\n{chunk.content}")

    return "\n\n".join(parts)


def build_messages(
    question: str,
    chunks: list[RetrievedChunk],
) -> list[dict]:
    """Build the OpenAI messages list for a RAG query.

    Args:
        question: User question string.
        chunks: Retrieved context chunks.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    system_text = load_template("rag_system")
    user_template = load_template("rag_user")

    context = format_context(chunks)
    # Security note (accepted risk): `question` is user-supplied text and
    # `context` contains PDF document content; both are interpolated directly
    # into the LLM prompt. This creates a structural prompt-injection surface
    # that cannot be fully eliminated in a RAG architecture without breaking
    # intended functionality. Mitigations in place:
    #   1. Question length is capped upstream in RAGPipeline.query().
    #   2. The system prompt explicitly instructs the model to ignore
    #      instructions embedded in the context or question.
    #   3. The LLM response is parsed through a strict Pydantic schema
    #      (RAGResponse) so injected content cannot reshape the output structure.
    # Additional hardening of the system prompt template is recommended.
    user_text = user_template.format(context=context, question=question)

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
