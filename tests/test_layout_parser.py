import unittest
from src.layout_parser import LayoutParser

class TestLayoutParser(unittest.TestCase):
    def setUp(self):
        self.parser = LayoutParser()

    def test_json_input(self):
        input_str = '[{"type": "text", "content": "test", "bbox": [0.1,0.1,0.9,0.9]}]'
        result = self.parser.parse(input_str)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['content'], 'test')

    def test_plain_text_input(self):
        result = self.parser.parse("add stars")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['content'], 'add stars')
        self.assertEqual(result[0]['bbox'], [0.7, 0.7, 1.0, 1.0])  # Default bottom-right

if __name__ == '__main__':
    unittest.main()