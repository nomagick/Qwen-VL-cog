# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, ConcatenateIterator
from transformers import AutoModel, AutoTokenizer

import patch_qwen
import tempfile
import shutil
import os

MODEL_PATH = './Qwen-VL-Chat'

import re

def add_prefix_to_images(text, prefix):
    pattern = r'<img>(.*?)</img>'

    def replace_function(match):
        matched_text = match.group(1)
        if matched_text.startswith("http://") or matched_text.startswith("https://"):
            return match.group(0)
        return f'<img>{os.path.join(prefix, matched_text.strip())}</img>'

    result = re.sub(pattern, replace_function, text)
    return result

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        model = patch_qwen.load_model_on_gpus(MODEL_PATH)
        model = patch_qwen.patch(model)

        self.model = model.eval()

    def predict(
        self,
        prompt: str = Input(
            description="Prompt for completion, in chatml format",
            default='''<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Given this image: <img>image1</img>, point out where the dog is<|im_end|>
<|im_start|>assistant
''',
        ),
        image1: Path = Input(description="Optional image you may use in your prompt known as image1", default=None),
        max_tokens: int = Input(
            description="Max new tokens to generate", default=2048, ge=1, le=32768
        ),
        temperature: float = Input(description="Temperature", default=0.75, ge=0, le=5),
        top_p: float = Input(description="Top_p", default=0.8, ge=0, le=1),
        image2: Path = Input(description="Optional image you may use in your prompt known as image2", default=None),
        image3: Path = Input(description="Optional image you may use in your prompt known as image3", default=None),
        files_archive: Path = Input(description="An archive of files you mentioned in your prompt if more images are needed", default=None),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""

        file_ns = str(tempfile.mkdtemp())
        try:
            if image1:
                shutil.copyfile(image1, os.path.join(file_ns,'image1'))
            if image2:
                shutil.copyfile(image2, os.path.join(file_ns,'image2'))
            if image3:
                shutil.copyfile(image3, os.path.join(file_ns,'image3'))

            if files_archive:
                shutil.unpack_archive(files_archive, file_ns, filter='data')

            yield from self.model.chat_stream_raw(
                self.tokenizer, add_prefix_to_images(prompt, file_ns), 
                max_new_tokens=max_tokens, 
                temperature=temperature, 
                top_p=top_p,
                max_window_size=32768
            )
        finally:
            shutil.rmtree(file_ns)
