import unittest

from tmvec import TMVEC_REPO
from tmvec.model import (TransformerEncoderModule,
                         TransformerEncoderModuleConfig)


class TestModelLoading(unittest.TestCase):
    def setUp(self):
        self.config = TransformerEncoderModuleConfig()
        self.model = TransformerEncoderModule(self.config)

    def test_from_hub(self):
        model = TransformerEncoderModule.from_pretrained(TMVEC_REPO)
        self.assertIsInstance(model, TransformerEncoderModule)

    def test_local_model(self):
        self.assertIsInstance(self.model, TransformerEncoderModule)

    def test_has_compile_ctx(self):
        # check if has attribute _compile_ctx needed for training
        # within lightning.Trainer
        self.assertTrue(hasattr(self.model, "_compiler_ctx"))

    def test_push_to_hub(self):
        # test if has method to push to hub and assert callable
        self.assertTrue(hasattr(self.model, "push_to_hub"))
        self.assertTrue(callable(self.model.push_to_hub))

    def config_json(self):
        # assert methods from_json_file and to_json_file
        self.assertTrue(hasattr(self.config, "to_json_file"))
        self.assertTrue(hasattr(self.config, "from_json_file"))
        self.assertTrue(callable(self.config.to_json_file))
        self.assertTrue(callable(self.config.from_json_file))
