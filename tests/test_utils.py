import unittest

from src.helpers.utils import chunk_text, rank_fusion


class UtilsTests(unittest.TestCase):
    def test_chunk_text_with_overlap(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
        self.assertEqual(chunks, ["abcdefghij", "ijklmnopqr", "qrstuvwxyz"])

    def test_rank_fusion_combines_lists(self) -> None:
        scores = rank_fusion(["a", "b", "c"], ["c", "b"])
        self.assertGreater(scores["b"], 0.0)
        self.assertGreater(scores["c"], 0.0)
        self.assertIn("a", scores)


if __name__ == "__main__":
    unittest.main()
