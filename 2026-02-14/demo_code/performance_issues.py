
def process_items(items):
    result = ""
    # Performance issue: string concatenation in loop
    for item in items:
    # TODO: Consider using list.append() and ''.join() for better performance
        result += str(item) + ", "
    
    # Magic numbers
    if len(items) > 42:
        return result
    
    return result[:100]  # Another magic number

def calculate_something():
    try:
        value = 10 / 0
    except:  # Bare except clause
        return None
    return value
