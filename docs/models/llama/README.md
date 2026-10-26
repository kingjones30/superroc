# Nanbeige4.3-33B-Base (`llama`) on Super ROC

Dense 33B Llama-architecture model (52 layers). Serve-only: `prompt` and
`chat` refuse llama models with a clear error (`superroc serve` instead).
Tested file: the published Q4_K_M GGUF
[Nanbeige4.3-33B-Base-Q4_K_M-GGUF](https://huggingface.co/nesymerp1/Nanbeige4.3-33B-Base-Q4_K_M-GGUF)
(`nanbeige4.3-33b-base-q4_k_m.gguf`). It is a base model: send plain text to `/v1/completions`.

## Serve

```sh
superroc serve --host 127.0.0.1 --port 8080 llm \
  -m /path/to/nanbeige4.3-33b-base-q4_k_m.gguf \
  --served-model-name nanbeige4.3-33b-base-superroc --context 32768 \
  -t 0.9 --top-p 0.6
```

## Supported file, quant types and limits

- Geometry: `head_dim` 128, GQA group (`heads / kv_heads`) at most 8.
  Other files fail at load instead of the first decode token.
- Projection tensors (`attn_q/k/v/output`, `ffn_gate/up/down`,
  `output.weight`): `Q8_0`, `Q4_K`, `Q3_K`, `Q5_K`, `Q6_K`.
  `F32`/`BF16`/`Q8_K` projections are refused at bind time: the batched
  prefill FP16 GEMM cannot run them.
- Norms are `F32`; `token_embd.weight` accepts `F32`, `BF16` and the K-quants.
- No sliding-window or scaled-RoPE files: a nonzero
  `llama.attention.sliding_window` or a `llama.rope.scaling.type` other than
  `none` is refused at load (full causal attention with unscaled RoPE only).
- Tokenizer: `tokenizer.ggml.model` must be `llama` with no pre-tokenizer or
  `pre = default`; the token array must match `llama.vocab_size` and the
  `bos`/`eos`/`padding` ids must be in range. Characters with no vocab piece
  fall back to `<unk>`.
- Chat: messages render as concatenated content (no chat template); tool
  requests are refused; structured-output instructions travel in the prompt.
  Incremental streamed text is not `Decode()`-exact.

## Speed and prompt cache

Ryzen AI Max+ 395, the file above, 128 greedy tokens after the prompt; llama.cpp ran the same file on the same machine.

| prompt | Super ROC prefill tok/s | llama.cpp prefill tok/s | Super ROC decode tok/s | llama.cpp decode tok/s |
| --- | --- | --- | --- | --- |
| 512 | 573.8 | 317.3 | 10.2 | 10.8 |
| 4,096 | 524.0 | 279.2 | 9.1 | 10.0 |

Prompt cache: a 3,041-token request that extends an earlier 3,001-token one reused all 3,001 cached tokens and
finished in 4.1 s instead of 9.3 s, with the same output as the uncached request.
