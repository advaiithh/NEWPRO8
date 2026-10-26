"""Spoken arithmetic with a safe AST evaluator (no eval)."""
from __future__ import annotations

import ast
import math
import operator as op
import re

from .words import words_to_digits

OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
       ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos}
FUNCS = {"sqrt": math.sqrt}


def safe_eval(expr: str) -> float:
    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 100:
                raise ValueError("exponent too large")
            return OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS and len(n.args) == 1:
            return FUNCS[n.func.id](ev(n.args[0]))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval"))


REWRITES = [
    (r"(\d+(?:\.\d+)?) percent of (\d+(?:\.\d+)?)", r"(\1/100)*\2"),
    (r"square root of (\d+(?:\.\d+)?)", r"sqrt(\1)"),
    (r"(\d+(?:\.\d+)?) squared", r"(\1**2)"),
    (r"(\d+(?:\.\d+)?) cubed", r"(\1**3)"),
    (r"(\d+(?:\.\d+)?) (?:to the power of|to the power|raised to) (\d+(?:\.\d+)?)", r"\1**\2"),
    (r"\b(?:plus|add(?:ed to)?)\b", "+"), (r"\b(?:minus|subtract(?:ed by)?|less)\b", "-"),
    (r"\b(?:times|multiplied by|multiply by|x)\b", "*"), (r"\b(?:divided by|over)\b", "/"),
    (r"\bmod(?:ulo)?\b", "%"),
]
TRIGGER = re.compile(r"\b(what(?:'s| is)|calculate|compute|how much is|whats|solve|work out)\b|\d\s*(?:\+|-|\*|/|x|plus|minus|times|divided)")


def _fmt(v: float) -> str:
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.4f}".rstrip("0").rstrip(".")


def match(text: str, ctx: object | None = None) -> str | None:
    t = words_to_digits(text.lower().replace("?", ""))
    if not TRIGGER.search(t) or not re.search(r"\d", t):
        return None
    t = re.sub(r"^(?:.*?)(?:what(?: is|'s)|whats|calculate|compute|how much is|solve|work out)\s+", "", t)
    t = re.sub(r"^(?:[a-z']+\s+){1,3}(?=\d)", "", t)          # ASR slips such as "what a 17 x 23"
    t = re.sub(r"\b(?:the|a|an)\b ?", "", t)
    for pat, rep in REWRITES:
        t = re.sub(pat, rep, t)
    expr = re.sub(r"[^0-9+\-*/().%a-z ]", "", t)
    if re.search(r"[a-z]", expr.replace("sqrt", "")):
        return None
    try:
        v = safe_eval(expr.strip())
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None
    return f"That is {_fmt(v)}."
