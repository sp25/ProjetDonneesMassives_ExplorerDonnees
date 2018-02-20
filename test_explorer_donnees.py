from unittest import TestCase
from explorer_donnees import explorer_donnees

class test_explorer_donnees(TestCase):
    def test_explorer_donnees(self):
        test = explorer_donnees()
        self.assertEqual(test, 1)
