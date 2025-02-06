import ast

def detect_pep_634(code, ast_root):
    matches = []

    class MatchFinder(ast.NodeVisitor):
        def visit_Match(self, node):
            start_line = node.lineno
            end_line = node.end_lineno  # Requires Python 3.10+ for end_lineno attribute
            match_code = code.splitlines()[start_line - 1:end_line]
            matches.append((start_line, end_line, "\n".join(match_code)))
            self.generic_visit(node)

    MatchFinder().visit(ast_root)

    return matches if matches else None
