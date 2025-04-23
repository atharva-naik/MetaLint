from State.SymbolTable import table


class Node:
    def __init__(self, name, expression=None, specifier=None):

        self.name = name
        self.expression = expression
        self.specifier = specifier

    def visit(self, context):
        value = None
        if self.expression:
            value = self.expression.visit(context)
        if context["type"] == "program":
            table.insertVariable(self.name, value, self.specifier)
            return table.getVariable(self.name)
        elif context["type"] == "struct":
            table.insertStructMember(context["name"], self.name, value, self.specifier)
            return value
        elif context["type"] == "routine":
            pass
        elif context["type"] == "object":
            pass