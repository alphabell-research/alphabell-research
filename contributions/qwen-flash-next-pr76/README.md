# Qwen3.8-Flash-Next PR 76: independent rendering check

Run a reproducible offline matrix against base `de4b8e4d43b917e7706784d8bb445c9af86a3540` and PR head `4381e064c58fa0d453f0630939b04140100ce0f7`.

The script downloads four public source files from their exact Hugging Face revisions, checks their SHA-256 hashes, then renders bounded synthetic conversations with Jinja 3.1.6. Downloaded sources live in a temporary directory that is removed when the script exits. It does not load a tokenizer, model weights or inference service, and does not execute tools. No credentials are required.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python qwen_pr76_review.py
```

To retain every case label and result:

```sh
.venv/bin/python qwen_pr76_review.py --output results.json
```

The expected denominator is 1,196 configuration cases:

| Check | Cases |
| --- | ---: |
| Previously accepted inputs remain byte-identical | 448 |
| Multiple leading system messages match an independently merged single-system control | 640 |
| Late-system rejection stays unchanged | 96 |
| Existing invalid-input rejection stays unchanged | 12 |

These are combinations of input shapes and settings, rather than distinct failure modes. The script also checks that each standalone template exactly matches its own tokenizer-config string and that all other tokenizer-config values remain unchanged. It exits nonzero if any matrix check fails.

Sources: [issue 75](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/discussions/75), [PR 76](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/discussions/76). The manifest records the exact source and script hashes.

This alphaBell check concerns text rendering and configuration consistency. Token IDs, server behavior, application compatibility and generation quality require separate validation.
