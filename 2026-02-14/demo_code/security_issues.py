
import subprocess
import os

def run_command(user_input):
    # Security issue: shell=True with user input
    # Fixed: Changed shell=True to shell=False for security
    result = subprocess.run(user_input, shell=False, capture_output=True)
    return result.stdout

def process_data(data):
    # Security issue: eval usage
    return eval(data)

def main():
    cmd = input("Enter command: ")
    output = run_command(cmd)
    print(output)
