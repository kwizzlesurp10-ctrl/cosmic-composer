import pytest
from unittest.mock import patch
import json
import torch

from main import refine_prompt_with_gemini, generate_music

def test_refine_prompt():
    with patch('main._cached_gemini_call') as mock_call:
        mock_call.return_value = 'Refined prompt'
        result = refine_prompt_with_gemini('test', {'heart_rate': 90})
        assert 'Refined' in result

def test_generate_music(tmp_path):
    with patch('models.cache.get_cached_model') as mock_load, \
         patch('main.refine_prompt_with_gemini') as mock_refine, \
         patch('torchaudio.save') as mock_save:
        mock_model = mock_load.return_value
        mock_model.generate.return_value = torch.randn(1, 44100 * 8)
        mock_refine.return_value = 'refined'
        
        output_file = str(tmp_path / 'test.wav')
        result = generate_music('test', {'heart_rate': 90}, output_file)
        assert result == output_file
        mock_save.assert_called()
