"""Download the six pretrained models used by the pipeline, pinned to the exact Hugging Face commits we used.

This is the only step that needs network access. The pipeline itself runs with HF_HUB_OFFLINE=1 and loads every model
from the local directory written here. No business data is downloaded: these are general-purpose language models,
all MIT or Apache-2.0 licensed and at most 4B parameters.

python download_models.py [out=models]   -> <out>/<model name>/ (config, tokenizer, safetensors weights)
"""
import os
import sys

from huggingface_hub import snapshot_download

# local name: (repository, commit, licence, parameters, role)
MODELS = {
    'multilingual-e5-small': ('intfloat/multilingual-e5-small', '614241f622f53c4eeff9890bdc4f31cfecc418b3',
                              'MIT', '118M', 'ANN candidate pass (frozen)'),
    'ms-marco-MiniLM-L12-v2': ('cross-encoder/ms-marco-MiniLM-L12-v2', '7b0235231ca2674cb8ca8f022859a6eba2b1c968',
                               'Apache-2.0', '33M', 'cross-encoder cev'),
    'mmarco-mMiniLMv2-L12-H384-v1': ('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
                                     '1427fd652930e4ba29e8149678df786c240d8825', 'Apache-2.0', '118M',
                                     'cross-encoder cem'),
    'bge-reranker-base': ('BAAI/bge-reranker-base', '2cfc18c9415c912f9d8155881c133215df768a70', 'MIT', '278M',
                          'cross-encoders ceb, cef'),
    'bge-reranker-v2-m3': ('BAAI/bge-reranker-v2-m3', '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e', 'Apache-2.0', '568M',
                           'cross-encoders cer (teacher only), cer2'),
    'Qwen3-Reranker-4B': ('Qwen/Qwen3-Reranker-4B', '22e683669bc0f0bd69640a1354a6d0aebcfeede5', 'Apache-2.0', '4.0B',
                          'cross-encoder ceq (LoRA)'),
}

if __name__ == '__main__':
    kv = dict(a.split('=', 1) for a in sys.argv[1:])
    out = kv.get('out', os.environ.get('ER_PRETRAINED', 'models'))
    for name, (repo, rev, lic, size, role) in MODELS.items():
        path = snapshot_download(repo, revision=rev, local_dir=f'{out}/{name}',
                                 allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'],
                                 ignore_patterns=['onnx/*', 'openvino/*'])
        print(f'{repo}@{rev[:8]} ({lic}, {size}; {role}) -> {path}', flush=True)
