import ast

def detect_pep_655(code: str, ast_root: ast.AST):
    occurrences = []
    
    # Walk through all nodes in the AST
    for node in ast.walk(ast_root):
        
        # Detect imports of Required and NotRequired
        if isinstance(node, ast.ImportFrom):
            if node.module == "typing":
                for alias in node.names:
                    if alias.name in {"Required", "NotRequired"}:
                        start_line = node.lineno
                        end_line = node.end_lineno
                        import_type = "import"
                        import_code = code.splitlines()[start_line - 1:end_line]
                        occurrences.append((start_line, end_line, import_type, "\n".join(import_code)))

        # Detect TypedDict definitions with Required/NotRequired fields
        elif isinstance(node, ast.ClassDef):
            is_typed_dict = any(
                isinstance(base, ast.Name) and base.id == "TypedDict" for base in node.bases
            )
            
            if is_typed_dict:
                # Search for assignments using Required or NotRequired in class body
                has_pep_655_fields = False
                for body_node in node.body:
                    if isinstance(body_node, ast.AnnAssign):
                        # Check if the type annotation uses Required or NotRequired
                        if isinstance(body_node.annotation, ast.Subscript) and isinstance(body_node.annotation.value, ast.Name):
                            if body_node.annotation.value.id in {"Required", "NotRequired"}:
                                has_pep_655_fields = True
                                break
                
                if has_pep_655_fields:
                    start_line = node.lineno
                    end_line = node.end_lineno
                    definition_type = "typed dict definition"
                    definition_code = code.splitlines()[start_line - 1:end_line]
                    occurrences.append((start_line, end_line, definition_type, "\n".join(definition_code)))

    return occurrences if occurrences else None

# main
if __name__ == "__main__":
    code = """
from typing import TypedDict, Required, NotRequired

class Movie(TypedDict):
    title: Required[str]
    director: NotRequired[str]

class Movie(TypedDict):
    title: str
    director: str"""

    ast_root = ast.parse(code)
    print(detect_pep_655(code, ast_root))
