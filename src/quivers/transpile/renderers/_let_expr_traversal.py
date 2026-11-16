"""Connect binder-free renderer IR expressions to the shared DSL traversal."""

from quivers.dsl.ast_nodes import LetExprNode
from quivers.transpile.ir import LetExprAffineMap


class _AffineExpressionExtension:
    """Expose affine operands while retaining their layout and source metadata."""

    def children(self, node: LetExprNode) -> tuple[LetExprNode, ...] | None:
        """Return affine operands in their declared source order."""
        if isinstance(node, LetExprAffineMap):
            return (node.weight, node.bias, *(source.value for source in node.sources))
        return None

    def rebuild(
        self, node: LetExprNode, children: tuple[LetExprNode, ...]
    ) -> LetExprNode:
        """Replace affine operands without changing the target-independent layout."""
        if not isinstance(node, LetExprAffineMap):
            raise TypeError(f"not an affine expression: {type(node).__name__}")
        weight, bias, *sources = children
        return node.with_(
            weight=weight,
            bias=bias,
            sources=tuple(
                source.with_(value=value)
                for source, value in zip(node.sources, sources, strict=True)
            ),
        )


AFFINE_EXPRESSION = _AffineExpressionExtension()
