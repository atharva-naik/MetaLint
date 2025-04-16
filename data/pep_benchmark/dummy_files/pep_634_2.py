import random
import json
import time


def handle_response(response):
    if isinstance(response, dict):
        if "error" in response:
            print(f"Error: {response['error']}")
        elif "data" in response:
            print(f"Data: {response['data']}")
        else:
            print("Unknown response format")
    elif isinstance(response, list):
        print("List of items:", response)
    else:
        print("Invalid response type")

def eval_expr(expr):
    if isinstance(expr, tuple):
        op, left, right = expr
        if op == '+':
            return eval_expr(left) + eval_expr(right)
        elif op == '*':
            return eval_expr(left) * eval_expr(right)
        else:
            raise ValueError(f"Unknown operator {op}")
    elif isinstance(expr, int):
        return expr
    else:
        raise TypeError("Invalid expression")

# Custom Exception
class APIError(Exception):
    pass

# Simulated backend that "processes" requests
def fake_api_call(payload: str) -> dict:
    try:
        data = json.loads(payload)
        expr = data.get("expression")
        if not expr:
            return {"error": "Missing expression field"}

        # Evaluate the expression
        result = eval_expr(tuple(expr))
        return {"data": {"result": result, "evaluated": True}}

    except (ValueError, TypeError) as e:
        return {"error": str(e)}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Higher-level logic simulating a pipeline
def process_request(expression: tuple):
    request = {
        "expression": expression
    }
    payload = json.dumps(request)
    response = fake_api_call(payload)
    handle_response(response)

# Generate complex expressions
def generate_random_expr(depth=3):
    if depth == 0:
        return random.randint(1, 10)
    op = random.choice(['+', '*'])
    return (op, generate_random_expr(depth - 1), generate_random_expr(depth - 1))

# Run batch of simulated requests
def run_pipeline(n=5):
    print("Processing requests...\n")
    for i in range(n):
        expr = generate_random_expr()
        print(f"[Request {i+1}] Expression: {expr}")
        process_request(expr)
        print("-" * 40)
        time.sleep(0.5)

if __name__ == "__main__":
    run_pipeline()