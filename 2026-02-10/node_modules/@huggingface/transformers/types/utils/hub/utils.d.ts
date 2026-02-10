/**
 * Joins multiple parts of a path into a single path, while handling leading and trailing slashes.
 *
 * @param {...string} parts Multiple parts of a path.
 * @returns {string} A string representing the joined path.
 */
export function pathJoin(...parts: string[]): string;
/**
 * Determines whether the given string is a valid URL.
 * @param {string|URL} string The string to test for validity as an URL.
 * @param {string[]} [protocols=null] A list of valid protocols. If specified, the protocol must be in this list.
 * @param {string[]} [validHosts=null] A list of valid hostnames. If specified, the URL's hostname must be in this list.
 * @returns {boolean} True if the string is a valid URL, false otherwise.
 */
export function isValidUrl(string: string | URL, protocols?: string[], validHosts?: string[]): boolean;
/**
 * Tests whether a string is a valid Hugging Face model ID or not.
 * Adapted from https://github.com/huggingface/huggingface_hub/blob/6378820ebb03f071988a96c7f3268f5bdf8f9449/src/huggingface_hub/utils/_validators.py#L119-L170
 *
 * @param {string} string The string to test
 * @returns {boolean} True if the string is a valid model ID, false otherwise.
 */
export function isValidHfModelId(string: string): boolean;
/**
 * Helper method to handle fatal errors that occur while trying to load a file from the Hugging Face Hub.
 * @param {number} status The HTTP status code of the error.
 * @param {string} remoteURL The URL of the file that could not be loaded.
 * @param {boolean} fatal Whether to raise an error if the file could not be loaded.
 * @returns {null} Returns `null` if `fatal = true`.
 * @throws {Error} If `fatal = false`.
 */
export function handleError(status: number, remoteURL: string, fatal: boolean): null;
/**
 * Read and track progress when reading a Response object
 *
 * @param {Response|import('./files.js').FileResponse} response The Response object to read
 * @param {(data: {progress: number, loaded: number, total: number}) => void} progress_callback The function to call with progress updates
 * @returns {Promise<Uint8Array>} A Promise that resolves with the Uint8Array buffer
 */
export function readResponse(response: Response | import("./files.js").FileResponse, progress_callback: (data: {
    progress: number;
    loaded: number;
    total: number;
}) => void): Promise<Uint8Array>;
//# sourceMappingURL=utils.d.ts.map