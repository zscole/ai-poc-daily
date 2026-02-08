#!/usr/bin/env node

/**
 * API Tester Script
 * Makes HTTP requests and displays formatted responses
 */

function parseArgs(args) {
  const result = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith('--')) {
      const key = args[i].slice(2);
      result[key] = args[i + 1];
      i++;
    }
  }
  return result;
}

async function makeRequest(method, url, data) {
  const options = {
    method: method.toUpperCase(),
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'User-Agent': 'AgentSkills-APITester/1.0',
    },
  };

  if (data && ['POST', 'PUT', 'PATCH'].includes(options.method)) {
    options.body = typeof data === 'string' ? data : JSON.stringify(data);
  }

  const startTime = Date.now();
  
  try {
    const response = await fetch(url, options);
    const duration = Date.now() - startTime;
    
    let body;
    const contentType = response.headers.get('content-type') || '';
    
    if (contentType.includes('application/json')) {
      body = await response.json();
    } else {
      body = await response.text();
    }

    return {
      success: true,
      status: response.status,
      statusText: response.statusText,
      headers: Object.fromEntries(response.headers.entries()),
      body,
      duration,
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
      duration: Date.now() - startTime,
    };
  }
}

function formatOutput(result) {
  const lines = [];
  
  lines.push('=== API Response ===');
  lines.push('');
  
  if (result.success) {
    lines.push(`Status: ${result.status} ${result.statusText}`);
    lines.push(`Duration: ${result.duration}ms`);
    lines.push('');
    
    lines.push('Headers:');
    for (const [key, value] of Object.entries(result.headers)) {
      lines.push(`  ${key}: ${value}`);
    }
    lines.push('');
    
    lines.push('Body:');
    if (typeof result.body === 'object') {
      lines.push(JSON.stringify(result.body, null, 2));
    } else {
      lines.push(result.body);
    }
  } else {
    lines.push(`Error: ${result.error}`);
    lines.push(`Duration: ${result.duration}ms`);
  }
  
  lines.push('');
  lines.push('=== End Response ===');
  
  return lines.join('\n');
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  
  const method = args.method || 'GET';
  const url = args.url;
  const data = args.data;
  
  if (!url) {
    console.error('Error: URL is required');
    console.error('Usage: request --url <url> [--method <method>] [--data <json>]');
    process.exit(1);
  }
  
  console.log(`Making ${method} request to ${url}...`);
  console.log('');
  
  const result = await makeRequest(method, url, data);
  console.log(formatOutput(result));
  
  process.exit(result.success ? 0 : 1);
}

main();
