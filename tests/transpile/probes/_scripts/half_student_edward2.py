"""Measure support, normalized densities, derivatives, and generated response draws."""

import json
from pathlib import Path
import sys

import edward2
import tensorflow as tf
import tensorflow_probability as tfp


def main():
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/io")
    namespace = {"edward2": edward2, "tf": tf, "tfp": tfp}
    exec((directory / "source.py").read_text(), namespace)  # noqa: S102
    model = namespace["model"]
    scale = tf.Variable(1.5, dtype=tf.float32)
    values = tf.constant([0.0, 0.5, 1.0])
    with tf.GradientTape() as tape:
        response = model(scale=scale, y=values)
        density = response.distribution.log_prob(values)
        score = tf.reduce_sum(density)
    gradient = tape.gradient(score, scale)
    negative = response.distribution.log_prob(tf.constant(-1.0))
    tf.random.set_seed(317)
    generated = model(scale=scale)
    samples = generated.distribution.sample(60000, seed=218)
    result = {
        "density": density.numpy().tolist(),
        "gradient": float(gradient.numpy()),
        "negative_is_impossible": bool(tf.math.is_inf(negative) & (negative < 0)),
        "generated_nonnegative": bool(tf.reduce_all(generated.value >= 0)),
        "sample_nonnegative": bool(tf.reduce_all(samples >= 0)),
        "sample_mean": float(tf.reduce_mean(samples)),
        "sample_cdf": [
            float(tf.reduce_mean(tf.cast(samples <= value, tf.float32)))
            for value in [0.25, 1.0, 3.0, 8.0]
        ],
    }
    (directory / "result.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
