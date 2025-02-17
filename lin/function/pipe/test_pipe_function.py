import unittest
import asyncio
from pipe_function.deepbricks_pipe_function import Pipe

class TestPipeModelFormatting(unittest.TestCase):
    def setUp(self):
        self.pipe = Pipe()
        self.pipe.valves.API_KEY = "sk-jRXDHkoPJ664Vz2iHnOREAL1pAeYCu6H1COOOkDFjj1914U0"
        
    def test_model_name_formatting(self):
        test_cases = [
            {
                "input": "deepbricksbycline.claude-3.5-sonnet",
                "expected": "claude-3.5-sonnet"
            },
            {
                "input": "claude-3.5-sonnet", 
                "expected": "claude-3.5-sonnet"
            },
            {
                "input": "deepbricksbycline.model-name",
                "expected": "model-name"
            }
        ]
        
        async def run_test(case):
            # Create mock body with model name
            body = {"model": case["input"]}
            
            # Call the pipe method and check payload
            try:
                await self.pipe.pipe(body, {})
            except Exception:
                pass  # We're only testing the model name formatting
            
            # Verify the formatted model name
            self.assertEqual(
                body["model"],
                case["expected"],
                f"Model name formatting failed for input: {case['input']}"
            )
            
        loop = asyncio.get_event_loop()
        for case in test_cases:
            with self.subTest(case=case):
                loop.run_until_complete(run_test(case))

if __name__ == "__main__":
    unittest.main()
