"""Small authored score-only models; standard library, no model service or pickle."""
import math
from statistics import mean


def _check(x, y=None):
    if not x or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 1 <= v <= 10 for v in x):
        raise ValueError("Expected finite raw scores on 1..10")
    if y is not None:
        if len(y) != len(x):
            raise ValueError("Feature and target lengths differ")
        _check(y)


def fit_score_model(kind, x, y, **options):
    _check(x, y)
    if kind == "raw":
        return {"kind": kind, "parameter_count": 0}
    if kind == "mean":
        return {"kind": kind, "value": mean(y), "parameter_count": 1}
    if kind == "affine":
        mx, my = mean(x), mean(y)
        variance = sum((v - mx) ** 2 for v in x)
        if variance <= 1e-12:
            raise ValueError("Affine raw-score variance is zero")
        slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / variance
        if slope <= 0:
            raise ValueError("Positive affine fit unavailable")
        return {"kind": kind, "slope": slope, "intercept": my - slope * mx, "parameter_count": 2}
    if kind == "quadratic_ridge":
        alpha = options["alpha"]
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alpha must be positive and finite")
        # Scale by the fixed published rating range; center features on this TRAIN fold only.
        z = [(v - 5.5) / 4.5 for v in x]
        q = [v * v for v in z]
        mz, mq, my = mean(z), mean(q), mean(y)
        u, v = [a - mz for a in z], [a - mq for a in q]
        aa = sum(a * a for a in u) + alpha
        bb = sum(a * a for a in v) + alpha
        ab = sum(a * b for a, b in zip(u, v))
        ay = sum(a * (b - my) for a, b in zip(u, y))
        by = sum(a * (b - my) for a, b in zip(v, y))
        determinant = aa * bb - ab * ab
        first, second = (ay * bb - by * ab) / determinant, (by * aa - ay * ab) / determinant
        return {"kind": kind, "alpha": alpha, "coefficients": [first, second],
                "intercept": my - first * mz - second * mq, "parameter_count": 3,
                "objective": "TRAIN sum squared error + alpha*(linear_coefficient^2 + quadratic_coefficient^2); unpenalized intercept"}
    if kind != "tree":
        raise ValueError("Unknown score model")
    depth, leaf = options["max_depth"], options["min_leaf"]
    if type(depth) is not int or not 1 <= depth <= 3 or type(leaf) is not int or leaf < 2:
        raise ValueError("Tree depth must be 1..3 and min_leaf >=2")

    def grow(indices, level):
        target = mean(y[i] for i in indices)
        node = {"value": target, "n": len(indices)}
        if level >= depth or len(indices) < 2 * leaf:
            return node
        values = sorted({x[i] for i in indices})
        parent_loss = sum((y[i] - target) ** 2 for i in indices)
        best = None
        for a, b in zip(values, values[1:]):
            threshold = (a + b) / 2
            left, right = [i for i in indices if x[i] <= threshold], [i for i in indices if x[i] > threshold]
            if min(len(left), len(right)) < leaf:
                continue
            ml, mr = mean(y[i] for i in left), mean(y[i] for i in right)
            loss = sum((y[i] - ml) ** 2 for i in left) + sum((y[i] - mr) ** 2 for i in right)
            if parent_loss - loss > 1e-12 and (best is None or loss < best[0] - 1e-12):
                best = (loss, threshold, left, right)
        if best is not None:
            node.update(threshold=best[1], left=grow(best[2], level + 1), right=grow(best[3], level + 1))
        return node

    tree = grow(list(range(len(x))), 0)
    def size(node):
        if "threshold" not in node:
            return 1, 0
        ll, ld = size(node["left"])
        rl, rd = size(node["right"])
        return ll + rl, 1 + max(ld, rd)
    leaves, actual_depth = size(tree)
    return {"kind": kind, "max_depth": depth, "min_leaf": leaf, "tree": tree,
            "leaves": leaves, "actual_depth": actual_depth, "parameter_count": 2 * leaves - 1}


def predict_score_model(model, x):
    _check(x)
    def one(value):
        kind = model["kind"]
        if kind == "raw":
            result = value
        elif kind == "mean":
            result = model["value"]
        elif kind == "affine":
            result = model["slope"] * value + model["intercept"]
        elif kind == "quadratic_ridge":
            z = (value - 5.5) / 4.5
            result = model["intercept"] + model["coefficients"][0] * z + model["coefficients"][1] * z * z
        elif kind == "tree":
            node = model["tree"]
            while "threshold" in node:
                node = node["left"] if value <= node["threshold"] else node["right"]
            result = node["value"]
        else:
            raise ValueError("Unknown score model")
        if not math.isfinite(result):
            raise ValueError("Nonfinite model output")
        return min(10.0, max(1.0, result))
    return [one(v) for v in x]


def score_metrics(predicted, targets):
    if not predicted or len(predicted) != len(targets):
        raise ValueError("Nonempty aligned predictions/targets required")
    errors = [a - b for a, b in zip(predicted, targets)]
    return {"n": len(errors), "mae": mean(abs(e) for e in errors),
            "rmse": math.sqrt(mean(e * e for e in errors)), "signed_bias": mean(errors)}
