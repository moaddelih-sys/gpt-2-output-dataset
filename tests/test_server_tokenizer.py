"""Request truncation regression tests using only the Python standard library.

Run: python -m unittest discover -s tests -p 'test_server_tokenizer.py' -v
"""
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch
from urllib.parse import quote


class TensorStub:
    """Keep token IDs and mask values inspectable without importing torch."""

    def __init__(self, values):
        self.values = values

    def unsqueeze(self, dim):
        if dim != 0:
            raise AssertionError('Expected a batch dimension at dim=0')
        return TensorStub([self.values])

    def to(self, device):
        return self


def load_server_module():
    # Scope dependency stubs to this import, just like the checkpoint tests.
    torch = ModuleType('torch')
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.tensor = Mock(side_effect=TensorStub)
    torch.ones_like = Mock(side_effect=lambda tensor: TensorStub(
        [[1] * len(row) for row in tensor.values]
    ))
    torch.no_grad = MagicMock()
    transformers = ModuleType('transformers')
    transformers.RobertaForSequenceClassification = Mock()
    transformers.RobertaTokenizer = Mock()
    for factory in (transformers.RobertaForSequenceClassification,
                    transformers.RobertaTokenizer):
        factory.from_pretrained.side_effect = AssertionError('No model downloads allowed')
    stubs = {'torch': torch, 'transformers': transformers, 'fire': ModuleType('fire')}
    path = Path(__file__).resolve().parents[1] / 'detector' / 'server.py'
    spec = importlib.util.spec_from_file_location('_server_tokenizer_tests', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class RequestTokenizerTests(unittest.TestCase):
    def assert_request(self, encoded_tokens, expected_tokens, model_max_length=8):
        server = load_server_module()
        # A plain namespace deliberately has no legacy max_len attribute; an
        # unrestricted Mock would silently create it and weaken this regression.
        server.tokenizer = SimpleNamespace(
            model_max_length=model_max_length,
            bos_token_id=0,
            eos_token_id=2,
            encode=Mock(return_value=list(encoded_tokens)),
        )
        self.assertFalse(hasattr(server.tokenizer, 'max_len'))
        server.device = 'cpu'

        probabilities = Mock()
        for method in ('detach', 'cpu', 'flatten', 'numpy'):
            getattr(probabilities, method).return_value = probabilities
        probabilities.tolist.return_value = [0.25, 0.75]
        logits = Mock()
        logits.softmax.return_value = probabilities
        server.model = Mock(return_value=(logits,))

        # Bypass BaseHTTPRequestHandler.__init__: run the real do_GET and
        # begin_content methods with an in-memory response, never an HTTP server.
        handler = object.__new__(server.RequestHandler)
        query = 'regression text + symbols & punctuation?'
        handler.path = '/?' + quote(query, safe='')
        handler.wfile = BytesIO()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        with patch('socket.socket', side_effect=AssertionError('No network allowed')):
            handler.do_GET()

        server.tokenizer.encode.assert_called_once_with(query)
        handler.send_response.assert_called_once_with(200)
        handler.send_header.assert_any_call('Content-Type', 'application/json;charset=UTF-8')
        handler.end_headers.assert_called_once_with()
        self.assertEqual(json.loads(handler.wfile.getvalue().decode('utf-8')), {
            'all_tokens': len(encoded_tokens),
            'used_tokens': len(expected_tokens),
            'fake_probability': 0.25,
            'real_probability': 0.75,
        })

        # Inspect what actually reached the model, not just the JSON counters.
        server.model.assert_called_once()
        args, kwargs = server.model.call_args
        self.assertEqual(len(args), 1)
        self.assertEqual(set(kwargs), {'attention_mask'})
        tokens = args[0]
        expected_input = [0] + expected_tokens + [2]
        self.assertEqual(tokens.values, [expected_input])
        self.assertLessEqual(len(tokens.values[0]), model_max_length)
        self.assertEqual(kwargs['attention_mask'].values, [[1] * len(expected_input)])
        server.torch.tensor.assert_called_once_with(expected_input)
        server.torch.ones_like.assert_called_once_with(tokens)
        server.torch.no_grad.assert_called_once_with()
        logits.softmax.assert_called_once_with(dim=-1)

    def test_long_query_uses_model_max_length_and_reserves_bos_eos(self):
        for limit, expected in (
            (2, []),
            (8, [10, 11, 12, 13, 14, 15]),
            (13, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        ):
            with self.subTest(model_max_length=limit):
                self.assert_request(list(range(10, 30)), expected, model_max_length=limit)

    def test_query_at_token_budget_is_not_truncated(self):
        self.assert_request(
            [10, 11, 12, 13, 14, 15],
            [10, 11, 12, 13, 14, 15],
        )

    def test_query_one_token_over_budget_drops_only_overflow(self):
        self.assert_request(
            [10, 11, 12, 13, 14, 15, 16],
            [10, 11, 12, 13, 14, 15],
        )

    def test_short_query_keeps_all_tokens_and_bos_eos(self):
        self.assert_request([10, 11, 12], [10, 11, 12])

    def test_nonempty_query_with_no_encoded_tokens_keeps_bos_eos(self):
        self.assert_request([], [])


if __name__ == '__main__':
    unittest.main()
