"""Checkpoint compatibility tests using only the Python standard library.

Run: python -m unittest discover -s tests -p 'test_server_checkpoint.py' -v
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


KNOWN_KEYS = (
    'roberta.embeddings.position_ids',
    'roberta.pooler.dense.weight',
    'roberta.pooler.dense.bias',
)


def load_server_module():
    # Import the real server with scoped stubs: no torch/Transformers installation,
    # model downloads, CUDA access, network sockets, or worker processes are needed.
    torch = ModuleType('torch')
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    transformers = ModuleType('transformers')
    transformers.RobertaForSequenceClassification = Mock()
    transformers.RobertaTokenizer = Mock()
    stubs = {'torch': torch, 'transformers': transformers, 'fire': ModuleType('fire')}
    path = Path(__file__).resolve().parents[1] / 'detector' / 'server.py'
    spec = importlib.util.spec_from_file_location('_server_checkpoint_tests', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def make_model(missing=(), unexpected=()):
    model = Mock()
    model.load_state_dict.return_value = SimpleNamespace(
        missing_keys=list(missing), unexpected_keys=list(unexpected)
    )
    return model


class CheckpointLoadingTests(unittest.TestCase):
    def setUp(self):
        self.server = load_server_module()
        self.state_dict = {'classifier.out_proj.weight': object()}

    def assert_loads(self, missing=(), unexpected=()):
        model = make_model(missing, unexpected)
        self.server._load_checkpoint_state_dict(model, self.state_dict)
        model.load_state_dict.assert_called_once_with(self.state_dict, strict=False)

    def prepare_main(self, model):
        self.server.torch.load = Mock(return_value={
            'args': {'large': False}, 'model_state_dict': self.state_dict,
        })
        self.server.RobertaForSequenceClassification.from_pretrained.return_value = model
        self.server.HTTPServer = Mock()
        self.server.subprocess = Mock()
        self.server.subprocess.check_output.return_value = b'0'
        self.server.serve_forever = Mock()
        self.server.print = Mock()

    def assert_startup_not_called(self, model):
        model.eval.assert_not_called()
        self.server.HTTPServer.assert_not_called()
        self.server.subprocess.check_output.assert_not_called()
        self.server.serve_forever.assert_not_called()

    def test_loads_matching_checkpoint_non_strictly(self):
        self.assert_loads()

    def test_accepts_each_known_missing_key(self):
        for key in KNOWN_KEYS:
            with self.subTest(key=key):
                self.assert_loads(missing=[key])

    def test_accepts_each_known_unexpected_key(self):
        for key in KNOWN_KEYS:
            with self.subTest(key=key):
                self.assert_loads(unexpected=[key])

    def test_accepts_issue_35_combination(self):
        self.assert_loads(missing=KNOWN_KEYS[:1], unexpected=KNOWN_KEYS[1:])

    def test_rejects_unknown_missing_key(self):
        key = 'classifier.out_proj.weight'
        model = make_model(missing=[key])
        with self.assertRaises(RuntimeError) as caught:
            self.server._load_checkpoint_state_dict(model, self.state_dict)
        self.assertIn("missing_keys=['" + key + "']", str(caught.exception))
        self.assertIn('unexpected_keys=[]', str(caught.exception))

    def test_rejects_unknown_unexpected_key(self):
        key = 'roberta.embeddings.token_type_ids'
        model = make_model(unexpected=[key])
        with self.assertRaises(RuntimeError) as caught:
            self.server._load_checkpoint_state_dict(model, self.state_dict)
        self.assertIn('missing_keys=[]', str(caught.exception))
        self.assertIn("unexpected_keys=['" + key + "']", str(caught.exception))

    def test_rejects_unknown_keys_mixed_with_known_mismatches(self):
        model = make_model(
            missing=[KNOWN_KEYS[0], 'classifier.out_proj.weight', 'classifier.out_proj.bias'],
            unexpected=[KNOWN_KEYS[1], KNOWN_KEYS[2], 'other.weight'],
        )
        with self.assertRaises(RuntimeError) as caught:
            self.server._load_checkpoint_state_dict(model, self.state_dict)
        self.assertEqual(
            str(caught.exception),
            'Incompatible checkpoint state_dict: '
            "missing_keys=['classifier.out_proj.bias', 'classifier.out_proj.weight'], "
            "unexpected_keys=['other.weight']",
        )

    def test_does_not_allow_key_prefixes_or_suffixes(self):
        for key in KNOWN_KEYS:
            for unknown in ('module.' + key, key + '.extra'):
                for field in ('missing', 'unexpected'):
                    with self.subTest(key=unknown, field=field):
                        model = make_model(**{field: [unknown]})
                        with self.assertRaises(RuntimeError) as caught:
                            self.server._load_checkpoint_state_dict(model, self.state_dict)
                        self.assertIn(unknown, str(caught.exception))

    def test_propagates_loader_errors(self):
        for error in (RuntimeError('size mismatch for classifier.out_proj.weight'),
                      ValueError('invalid state_dict')):
            with self.subTest(error=error):
                model = make_model()
                model.load_state_dict.side_effect = error
                with self.assertRaises(type(error)) as caught:
                    self.server._load_checkpoint_state_dict(model, self.state_dict)
                self.assertIs(caught.exception, error)
                model.load_state_dict.assert_called_once_with(self.state_dict, strict=False)

    def test_main_accepts_known_mismatches(self):
        model = make_model(missing=KNOWN_KEYS[:1], unexpected=KNOWN_KEYS[1:])
        self.prepare_main(model)
        self.server.main('fake-checkpoint.pt', port=8123, device='cpu')
        self.server.torch.load.assert_called_once_with('fake-checkpoint.pt', map_location='cpu')
        model.load_state_dict.assert_called_once_with(self.state_dict, strict=False)
        model.eval.assert_called_once_with()
        self.server.RobertaForSequenceClassification.from_pretrained.assert_called_once_with('roberta-base')
        self.server.RobertaTokenizer.from_pretrained.assert_called_once_with('roberta-base')
        self.server.HTTPServer.assert_called_once_with(('0.0.0.0', 8123), self.server.RequestHandler)
        self.server.serve_forever.assert_called_once_with(
            self.server.HTTPServer.return_value, model,
            self.server.RobertaTokenizer.from_pretrained.return_value, 'cpu',
        )

    def test_main_aborts_before_eval_or_server_on_unknown_keys(self):
        for field in ('missing', 'unexpected'):
            with self.subTest(field=field):
                model = make_model(**{field: ['classifier.out_proj.weight']})
                self.prepare_main(model)
                with self.assertRaises(RuntimeError):
                    self.server.main('fake-checkpoint.pt', device='cpu')
                model.load_state_dict.assert_called_once_with(self.state_dict, strict=False)
                self.assert_startup_not_called(model)

    def test_main_aborts_when_loader_raises(self):
        model = make_model()
        error = RuntimeError('size mismatch for classifier.out_proj.weight')
        model.load_state_dict.side_effect = error
        self.prepare_main(model)
        with self.assertRaises(RuntimeError) as caught:
            self.server.main('fake-checkpoint.pt', device='cpu')
        self.assertIs(caught.exception, error)
        self.assert_startup_not_called(model)


if __name__ == '__main__':
    unittest.main()
