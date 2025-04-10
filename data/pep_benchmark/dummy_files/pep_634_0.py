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