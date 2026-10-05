import unittest

from name_tools import normalize_name


class NormalizeNameTests(unittest.TestCase):
    def test_collapses_whitespace(self):
        self.assertEqual(normalize_name("  Ada   Lovelace "), "ada lovelace")

    def test_unicode_casefold(self):
        self.assertEqual(normalize_name("Straße"), "strasse")

    def test_tabs_and_newlines(self):
        self.assertEqual(normalize_name("Grace\t\nHopper"), "grace hopper")

    def test_empty_name(self):
        self.assertEqual(normalize_name(" \t\n"), "")


if __name__ == "__main__":
    unittest.main()
