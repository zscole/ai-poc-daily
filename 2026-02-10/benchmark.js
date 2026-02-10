/**
 * Transformers.js v4 Benchmark
 * 
 * Measures embedding throughput and latency
 * Compare with v3 to see the 4x improvement from new ONNX Runtime
 */

import { pipeline, env } from '@huggingface/transformers';

// Configure for Node.js environment
env.allowLocalModels = true;
env.useBrowserCache = false;
env.cacheDir = './.cache';

// Benchmark configuration
const WARMUP_RUNS = 5;
const BENCHMARK_RUNS = 50;
const TEST_SENTENCES = [
  "The quick brown fox jumps over the lazy dog.",
  "Artificial intelligence is transforming how we build software.",
  "WebGPU enables hardware-accelerated graphics and compute in browsers.",
  "Vector embeddings capture semantic meaning in dense numerical form.",
  "Edge computing brings AI inference closer to the data source."
];

/**
 * Run benchmark and collect statistics
 */
async function runBenchmark() {
  console.log('Transformers.js v4 Benchmark');
  console.log('='.repeat(50));
  console.log();
  
  // Load model
  console.log('Loading model: Xenova/all-MiniLM-L6-v2');
  const loadStart = performance.now();
  const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    dtype: 'fp32',
    device: 'auto'
  });
  const loadTime = performance.now() - loadStart;
  console.log(`Model loaded in ${(loadTime / 1000).toFixed(2)}s`);
  console.log();
  
  // Warmup
  console.log(`Warmup (${WARMUP_RUNS} runs)...`);
  for (let i = 0; i < WARMUP_RUNS; i++) {
    await embedder(TEST_SENTENCES[i % TEST_SENTENCES.length], { pooling: 'mean', normalize: true });
  }
  console.log('Warmup complete');
  console.log();
  
  // Benchmark single inference
  console.log(`Benchmarking single inference (${BENCHMARK_RUNS} runs)...`);
  const singleTimes = [];
  
  for (let i = 0; i < BENCHMARK_RUNS; i++) {
    const sentence = TEST_SENTENCES[i % TEST_SENTENCES.length];
    const start = performance.now();
    await embedder(sentence, { pooling: 'mean', normalize: true });
    singleTimes.push(performance.now() - start);
  }
  
  // Benchmark batch inference
  console.log('Benchmarking batch inference (5 sentences)...');
  const batchTimes = [];
  
  for (let i = 0; i < BENCHMARK_RUNS; i++) {
    const start = performance.now();
    for (const sentence of TEST_SENTENCES) {
      await embedder(sentence, { pooling: 'mean', normalize: true });
    }
    batchTimes.push(performance.now() - start);
  }
  
  // Calculate statistics
  const calcStats = (times) => {
    times.sort((a, b) => a - b);
    const sum = times.reduce((a, b) => a + b, 0);
    return {
      mean: sum / times.length,
      median: times[Math.floor(times.length / 2)],
      p95: times[Math.floor(times.length * 0.95)],
      min: times[0],
      max: times[times.length - 1]
    };
  };
  
  const singleStats = calcStats(singleTimes);
  const batchStats = calcStats(batchTimes);
  
  // Print results
  console.log();
  console.log('Results');
  console.log('-'.repeat(50));
  console.log();
  console.log('Single Inference Latency:');
  console.log(`  Mean:   ${singleStats.mean.toFixed(1)}ms`);
  console.log(`  Median: ${singleStats.median.toFixed(1)}ms`);
  console.log(`  P95:    ${singleStats.p95.toFixed(1)}ms`);
  console.log(`  Min:    ${singleStats.min.toFixed(1)}ms`);
  console.log(`  Max:    ${singleStats.max.toFixed(1)}ms`);
  console.log();
  console.log('Batch Inference (5 sentences):');
  console.log(`  Mean:   ${batchStats.mean.toFixed(1)}ms`);
  console.log(`  Median: ${batchStats.median.toFixed(1)}ms`);
  console.log(`  P95:    ${batchStats.p95.toFixed(1)}ms`);
  console.log();
  console.log('Throughput:');
  console.log(`  Single: ${(1000 / singleStats.mean).toFixed(1)} embeddings/sec`);
  console.log(`  Batch:  ${(5000 / batchStats.mean).toFixed(1)} embeddings/sec`);
  console.log();
  
  // Context for the numbers
  console.log('Context:');
  console.log('  - Transformers.js v4 achieves ~4x speedup over v3 for BERT models');
  console.log('  - MultiHeadAttention operator provides fused attention computation');
  console.log('  - Same code runs in browser, Node.js, Bun, and Deno');
  console.log('  - WebGPU enables GPU acceleration when available');
  console.log();
}

runBenchmark().catch(console.error);
