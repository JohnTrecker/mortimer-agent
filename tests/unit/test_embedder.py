"""Unit tests for the Embedder class."""


class TestEmbedder:
    def test_embedder_instantiates(self):
        from mortimer.retrieval.embedder import Embedder

        embedder = Embedder("all-MiniLM-L6-v2")
        assert embedder is not None

    def test_embed_texts_returns_list(self):
        from mortimer.retrieval.embedder import Embedder

        embedder = Embedder("all-MiniLM-L6-v2")
        result = embedder.embed_texts(["Hello world", "Test sentence"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_embed_texts_each_is_list_of_floats(self):
        from mortimer.retrieval.embedder import Embedder

        embedder = Embedder("all-MiniLM-L6-v2")
        result = embedder.embed_texts(["Hello"])
        assert isinstance(result[0], list)
        assert all(isinstance(v, float) for v in result[0])

    def test_embed_texts_consistent_dimension(self):
        from mortimer.retrieval.embedder import Embedder

        embedder = Embedder("all-MiniLM-L6-v2")
        result = embedder.embed_texts(["sentence one", "sentence two", "sentence three"])
        dims = [len(v) for v in result]
        assert len(set(dims)) == 1

    def test_embed_query_returns_list_of_floats(self):
        from mortimer.retrieval.embedder import Embedder

        embedder = Embedder("all-MiniLM-L6-v2")
        result = embedder.embed_query("What is AI?")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_embed_query_same_dimension_as_embed_texts(self):
        from mortimer.retrieval.embedder import Embedder

        embedder = Embedder("all-MiniLM-L6-v2")
        text_embed = embedder.embed_texts(["Test sentence"])[0]
        query_embed = embedder.embed_query("Test sentence")
        assert len(text_embed) == len(query_embed)

    def test_embed_texts_empty_list_returns_empty(self):
        from mortimer.retrieval.embedder import Embedder

        embedder = Embedder("all-MiniLM-L6-v2")
        result = embedder.embed_texts([])
        assert result == []

    def test_embed_texts_special_characters(self):
        from mortimer.retrieval.embedder import Embedder

        embedder = Embedder("all-MiniLM-L6-v2")
        result = embedder.embed_texts(["Unicode: \u4e2d\u6587 \U0001f600"])
        assert len(result) == 1
        assert len(result[0]) > 0

    def test_embed_texts_does_not_mutate_input(self):
        from mortimer.retrieval.embedder import Embedder

        embedder = Embedder("all-MiniLM-L6-v2")
        original = ["hello", "world"]
        copy = list(original)
        embedder.embed_texts(original)
        assert original == copy
