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