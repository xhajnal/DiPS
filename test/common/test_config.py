import unittest
from common.config import *


class MyTestCase(unittest.TestCase):
    def test_parse_numbers(self):
        a = load_config()
        print(a)
        ## TODO
        pass


if __name__ == '__main__':
    unittest.main()