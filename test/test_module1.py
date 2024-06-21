# test_module1.py

import unittest
from projectProject.module1 import say_hello

class TestModule1(unittest.TestCase):

    def test_say_hello(self):
        self.assertEqual(say_hello("World"), "Hello, World!")

if __name__ == '__main__':
    unittest.main()
