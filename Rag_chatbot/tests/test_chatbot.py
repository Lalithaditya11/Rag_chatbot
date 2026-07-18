import io
import unittest
from contextlib import redirect_stdout

import chatbot


class ChatbotTests(unittest.TestCase):
    def test_main_accepts_question_argument(self):
        output = io.StringIO()
        with redirect_stdout(output):
            chatbot.main(["what is rag"])
        text = output.getvalue()
        self.assertIn("rag", text.lower())


if __name__ == "__main__":
    unittest.main()
