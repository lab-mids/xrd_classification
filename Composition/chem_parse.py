import re
from collections import defaultdict
from pymatgen.core import Composition
def parse_formula(comp: str):
    """
    Parse chemical formula with support for parentheses, decimals,
    multi-digit counts, and nested groups.
    Returns {element: amount}.
    """
    token_pattern = r'([A-Z][a-z]?|\(|\)|\d*\.\d+|\d+)'
    tokens = re.findall(token_pattern, comp)

    stack = [defaultdict(float)]
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == "(":
            stack.append(defaultdict(float))

        elif token == ")":
            group = stack.pop()
            i += 1

            # multiplier after ')'
            if i < len(tokens) and re.match(r'\d|\d*\.\d+', tokens[i]):
                mult = float(tokens[i])
            else:
                mult = 1.0
                i -= 1

            for el, amt in group.items():
                stack[-1][el] += amt * mult

        elif re.match(r'[A-Z][a-z]?', token):
            el = token
            i += 1

            if i < len(tokens) and re.match(r'\d|\d*\.\d+', tokens[i]):
                amt = float(tokens[i])
            else:
                amt = 1.0
                i -= 1

            stack[-1][el] += amt

        i += 1

    return dict(stack.pop())


def clean_formula_chemically_no_space(formula):
    try:
        comp = Composition(formula)

        expanded = comp.get_el_amt_dict()

        # concatenated string with no spaces
        return ''.join(f"{el}{int(amount)}" for el, amount in expanded.items())

    except Exception as e:
        print(f"Could not parse {formula}: {e}")
        return formula



# ============================================================
# Chemical formula normalization
# ============================================================
def normalize_formula(formula: str) -> str:
    try:
        comp = Composition(formula)
        el_amt = comp.get_el_amt_dict()
        return "".join(f"{el}{amt:.4f}" for el, amt in el_amt.items())
    except Exception:
        return formula
