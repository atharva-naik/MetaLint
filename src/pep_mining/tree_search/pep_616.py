import ast

def detect_pep_616(code: str, ast_root: ast.Module):
    
    # List to store lines with removeprefix or removesuffix calls
    occurrences = []

    # Custom visitor to find calls to removeprefix or removesuffix
    class PrefixSuffixVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            # Check if the function is an attribute call
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
                # Check if the attribute is 'removeprefix' or 'removesuffix'
                if func_name in {"removeprefix", "removesuffix"}:
                    line_number = node.lineno
                    line_content = code.splitlines()[line_number - 1].strip()
                    occurrences.append((line_number, line_content))
            self.generic_visit(node)

    # Visit the AST to find function calls
    PrefixSuffixVisitor().visit(ast_root)
    
    return occurrences if occurrences else None

if __name__ == "__main__":
    # Example usage:
    code = """
    s = "prefix_value_suffix"
    s1 = s.removeprefix("prefix_")
    s2 = s.removesuffix("_suffix")
    s1 = s.removeprefix("prefix_")
    s2 = s.removesuffix("_suffix")
    """

    print(detect_pep_616(code))
