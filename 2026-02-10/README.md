# Transformers.js v4 WebGPU Demo - Semantic Search

**Released: February 10, 2026**

Demonstrates the major Transformers.js v4 release (Feb 9, 2026) with a semantic search implementation that runs identically in browsers AND Node.js/Bun/Deno.

## Why This Matters

Transformers.js v4 is a significant milestone for browser-side AI. After nearly a year of development, it brings:

1. **WebGPU Runtime in JavaScript Runtimes** - The same WebGPU-accelerated code now runs in Node.js, Bun, and Deno, not just browsers. Write once, run everywhere.

2. **4x Speedup for BERT Models** - New ONNX Runtime with fused attention operations (MultiHeadAttention operator) dramatically improves embedding performance.

3. **8B+ Parameter Model Support** - Tested with GPT-OSS 20B (q4f16) at ~60 tokens/second on M4 Pro Max.

4. **Standalone Tokenizers Library** - The new @huggingface/tokenizers package is only 8.8kB gzipped with zero dependencies.

5. **Full Offline Support** - Models are cached locally after first download, enabling true offline AI applications.

## Quick Start

### Node.js

```bash
npm install
npm run demo
```

### Browser

```bash
npm run serve
# Open http://localhost:3000
```

Or simply open `index.html` directly - it uses CDN imports.

### Benchmark

```bash
npm run benchmark
```

## What This Demo Does

1. Loads a 22M parameter embedding model (all-MiniLM-L6-v2)
2. Generates 384-dimensional embeddings for 15 sample documents
3. Accepts natural language queries
4. Finds semantically similar documents using cosine similarity
5. Reports latency and throughput metrics

## Architecture

```
Query: "How can I run AI models in a web browser?"
  |
  v
[Transformers.js v4 Pipeline]
  |
  +-- Tokenizer (8.8kB standalone library)
  |
  +-- ONNX Runtime (WebGPU or WASM)
  |
  +-- Mean Pooling + L2 Normalization
  |
  v
Query Embedding (384 dimensions)
  |
  v
[Cosine Similarity vs Document Embeddings]
  |
  v
Top-K Results
```

## Key Files

- `index.js` - Node.js semantic search demo
- `index.html` - Browser version (same logic, different UI)
- `benchmark.js` - Performance measurement script
- `package.json` - Dependencies (@huggingface/transformers@next)

## Performance

Typical results on Apple M-series:

| Metric | Value |
|--------|-------|
| Model Load | 2-5s (first run), <1s (cached) |
| Single Embedding | 15-40ms |
| Batch (15 docs) | 200-400ms |
| Throughput | 25-60 embeddings/sec |

WebGPU provides additional acceleration when available.

## Transformers.js v4 Features Used

- `pipeline('feature-extraction', ...)` - High-level API
- `device: 'auto'` - Automatic WebGPU/WASM selection
- `pooling: 'mean'` - Mean pooling for sentence embeddings
- `normalize: true` - L2 normalization for cosine similarity

## Context: Why Transformers.js v4 is Hot

The release timing is significant:

1. **Agentic AI Explosion** - OpenAI and Anthropic both released coding agents this week. Running AI at the edge becomes crucial for latency-sensitive agent tasks.

2. **Browser as Platform** - Apple's Xcode now integrates AI agents. The browser is the universal deployment target.

3. **Cost Pressure** - API costs add up. Local inference with Transformers.js is free after the initial download.

4. **Privacy** - Data never leaves the device. Critical for enterprise adoption.

## Related News (Feb 9-10, 2026)

- Transformers.js v4 announced on HuggingFace Blog
- Anthropic Opus 4.6 with "agent teams" released
- OpenAI and Anthropic both drop agentic coding models
- Apple Xcode adds OpenAI/Anthropic integration
- Claude's C Compiler (CCC) compiles Linux kernel C files

## License

MIT

## References

- [Transformers.js v4 Blog Post](https://huggingface.co/blog/transformersjs-v4)
- [Transformers.js GitHub](https://github.com/huggingface/transformers.js)
- [Transformers.js Examples](https://github.com/huggingface/transformers.js-examples)
- [ONNX Runtime WebGPU](https://onnxruntime.ai/)
