/**
 * Loads and caches the WASM binary for ONNX Runtime.
 * @param {string} wasmURL The URL of the WASM file to load.
 * @returns {Promise<ArrayBuffer|null>} The WASM binary as an ArrayBuffer, or null if loading failed.
 */
export function loadWasmBinary(wasmURL: string): Promise<ArrayBuffer | null>;
/**
 * Loads and caches the WASM Factory for ONNX Runtime.
 * @param {string} libURL The URL of the WASM Factory to load.
 * @returns {Promise<string|null>} The blob URL of the WASM Factory, or null if loading failed.
 */
export function loadWasmFactory(libURL: string): Promise<string | null>;
/**
 * Checks if the given URL is a blob URL (created via URL.createObjectURL).
 * Blob URLs should not be cached as they are temporary in-memory references.
 * @param {string} url - The URL to check.
 * @returns {boolean} True if the URL is a blob URL, false otherwise.
 */
export function isBlobURL(url: string): boolean;
/**
 * Converts any URL to an absolute URL if needed.
 * If the URL is already absolute (http://, https://, or blob:), returns it unchanged (handled by new URL(...)).
 * Otherwise, resolves it relative to the current page location (browser) or module location (Node/Bun/Deno).
 * @param {string} url - The URL to convert (can be relative or absolute).
 * @returns {string} The absolute URL.
 */
export function toAbsoluteURL(url: string): string;
//# sourceMappingURL=cacheWasm.d.ts.map