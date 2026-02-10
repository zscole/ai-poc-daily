/**
 * Transformers.js v4 WebGPU Demo - Semantic Search
 * 
 * Demonstrates the new Transformers.js v4 features:
 * - WebGPU acceleration in Node.js (not just browsers anymore)
 * - Same code runs in browser and server
 * - 4x faster BERT embeddings with new ONNX Runtime
 * - Full offline support after initial download
 * 
 * Released: February 9, 2026
 * https://huggingface.co/blog/transformersjs-v4
 */

import { pipeline, env } from '@huggingface/transformers';

// Configure for Node.js environment
env.allowLocalModels = true;
env.useBrowserCache = false; // Disable browser cache in Node.js
env.cacheDir = './.cache'; // Use local directory for model cache

// Sample documents for semantic search
const DOCUMENTS = [
  "WebGPU is a new web standard for GPU acceleration in browsers.",
  "Transformers.js v4 brings WebGPU support to Node.js and Deno.",
  "ONNX Runtime's new C++ WebGPU backend enables cross-platform AI.",
  "Semantic search uses embeddings to find conceptually similar text.",
  "Large language models can now run entirely in the browser.",
  "Edge AI reduces latency by processing data locally.",
  "The new tokenizers library is only 8.8kB gzipped.",
  "BERT models achieve 4x speedup with MultiHeadAttention operators.",
  "Vector databases store embeddings for similarity search.",
  "Fine-tuning allows models to specialize for specific domains.",
  "Mixture of Experts architectures improve model efficiency.",
  "State-space models like Mamba offer alternatives to attention.",
  "Quantization reduces model size while preserving accuracy.",
  "WebAssembly enables running compiled code in browsers.",
  "The ONNX format provides interoperability between frameworks."
];

/**
 * Compute cosine similarity between two vectors
 */
function cosineSimilarity(a, b) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Format duration in human-readable format
 */
function formatDuration(ms) {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

/**
 * Main semantic search demo
 */
async function main() {
  console.log('='.repeat(60));
  console.log('Transformers.js v4 WebGPU Demo - Semantic Search');
  console.log('='.repeat(60));
  console.log();
  
  // Check WebGPU availability
  const hasWebGPU = typeof navigator !== 'undefined' && navigator.gpu;
  console.log(`Runtime: Node.js ${process.version}`);
  console.log(`WebGPU: ${hasWebGPU ? 'Available' : 'Not available (using WASM fallback)'}`);
  console.log();
  
  // Load the embedding model
  console.log('Loading embedding model...');
  console.log('Model: Xenova/all-MiniLM-L6-v2 (22M params, 384 dimensions)');
  console.log();
  
  const loadStart = performance.now();
  const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    dtype: 'fp32',
    device: 'auto' // Uses WebGPU if available, falls back to WASM
  });
  const loadTime = performance.now() - loadStart;
  
  console.log(`Model loaded in ${formatDuration(loadTime)}`);
  console.log();
  
  // Generate embeddings for all documents
  console.log(`Embedding ${DOCUMENTS.length} documents...`);
  const embedStart = performance.now();
  
  const documentEmbeddings = [];
  for (const doc of DOCUMENTS) {
    const output = await embedder(doc, { pooling: 'mean', normalize: true });
    documentEmbeddings.push(Array.from(output.data));
  }
  
  const embedTime = performance.now() - embedStart;
  const avgTime = embedTime / DOCUMENTS.length;
  
  console.log(`Embedded ${DOCUMENTS.length} documents in ${formatDuration(embedTime)}`);
  console.log(`Average: ${formatDuration(avgTime)} per document`);
  console.log();
  
  // Interactive search demo
  const queries = [
    "How can I run AI models in a web browser?",
    "What makes transformers.js v4 faster?",
    "Tell me about efficient model architectures"
  ];
  
  console.log('-'.repeat(60));
  console.log('Semantic Search Results');
  console.log('-'.repeat(60));
  console.log();
  
  for (const query of queries) {
    console.log(`Query: "${query}"`);
    console.log();
    
    // Embed the query
    const queryStart = performance.now();
    const queryOutput = await embedder(query, { pooling: 'mean', normalize: true });
    const queryEmbedding = Array.from(queryOutput.data);
    const queryTime = performance.now() - queryStart;
    
    // Calculate similarities
    const similarities = documentEmbeddings.map((docEmb, idx) => ({
      index: idx,
      document: DOCUMENTS[idx],
      similarity: cosineSimilarity(queryEmbedding, docEmb)
    }));
    
    // Sort by similarity and get top 3
    similarities.sort((a, b) => b.similarity - a.similarity);
    const topResults = similarities.slice(0, 3);
    
    console.log(`  Search time: ${formatDuration(queryTime)}`);
    console.log('  Top matches:');
    for (let i = 0; i < topResults.length; i++) {
      const r = topResults[i];
      console.log(`    ${i + 1}. [${(r.similarity * 100).toFixed(1)}%] ${r.document}`);
    }
    console.log();
  }
  
  // Performance summary
  console.log('='.repeat(60));
  console.log('Performance Summary');
  console.log('='.repeat(60));
  console.log();
  console.log(`  Model load time:     ${formatDuration(loadTime)}`);
  console.log(`  Batch embedding:     ${formatDuration(embedTime)} (${DOCUMENTS.length} docs)`);
  console.log(`  Avg per document:    ${formatDuration(avgTime)}`);
  console.log(`  Embedding dimension: 384`);
  console.log();
  console.log('Key Transformers.js v4 Features Demonstrated:');
  console.log('  - WebGPU/WASM automatic device selection');
  console.log('  - Streaming model download with caching');
  console.log('  - Same code works in Node.js, Bun, Deno, and browsers');
  console.log('  - Mean pooling with L2 normalization');
  console.log();
}

main().catch(console.error);
