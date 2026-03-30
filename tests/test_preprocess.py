import unittest
from preprocess import (
    to_lower,
    remove_punctuation,
    remove_numbers,
    strip_whitespace,
    tokenize,
    preprocess,
)

class TestPreprocess(unittest.TestCase):
    def test_to_lower(self):
        self.assertEqual(to_lower("HeLLo"), "hello")

    def test_remove_punctuation(self):
        self.assertEqual(remove_punctuation("Hello, world!"), "Hello world")

    def test_remove_numbers(self):
        self.assertEqual(remove_numbers("abc123def"), "abcdef")

    def test_strip_whitespace(self):
        self.assertEqual(strip_whitespace("  This   is   spaced  "), "This is spaced")

    def test_tokenize(self):
        self.assertEqual(tokenize("one two three"), ["one", "two", "three"])

    def test_preprocess(self):
        raw = "  Hello, WORLD! 123   "
        cleaned = preprocess(raw)
        self.assertEqual(cleaned, "hello world")

if __name__ == "__main__":
    unittest.main()
