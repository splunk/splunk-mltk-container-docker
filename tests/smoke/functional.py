"""Target-specific functional checks (runs inside the container, offline).

Usage: python - <target> < functional.py
Prints PASS/FAIL lines; exits 1 if any check fails.
"""

import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
target = sys.argv[1]
failures = 0


def check(name, func):
    global failures
    try:
        detail = func()
        print(f"PASS {name}" + (f": {detail}" if detail else ""), flush=True)
    except BaseException as error:  # noqa: BLE001 - report every failure kind
        failures += 1
        print(f"FAIL {name}: {traceback.format_exception_only(type(error), error)[-1].strip()[:300]}", flush=True)


def torch_compute():
    import torch
    x = torch.ones(64, 64)
    assert float((x @ x).sum()) == 64 * 64 * 64
    return f"torch {torch.__version__}, cuda build {torch.version.cuda}"


def tensorflow_compute():
    import tensorflow as tf
    x = tf.ones((64, 64))
    assert float(tf.reduce_sum(tf.matmul(x, x))) == 64 * 64 * 64
    return f"tensorflow {tf.__version__}"


def sklearn_fit():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    X = np.random.default_rng(0).normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(int)
    assert LogisticRegression().fit(X, y).score(X, y) > 0.9


def escu_torch_models():
    import torch
    base = "/dltk/app/model/data"
    names = [n for n in os.listdir(base) if n.endswith(".pt")]
    for name in names:
        for path in (os.path.join(base, name), os.path.join(base, name[:-3], name)):
            state = torch.load(path, map_location="cpu", weights_only=True)
            assert isinstance(state, dict) and state, path
    return f"{len(names)} models, flat and per-model layout"


def escu_dga_savedmodel():
    import tf_keras
    model = tf_keras.models.load_model("/dltk/app/model/data/pretrained_dga_model_dsdl")
    return f"{type(model).__name__}"


def spacy_model():
    import spacy
    doc = spacy.load("en_core_web_sm")("Splunk is based in San Francisco.")
    return f"entities {[e.label_ for e in doc.ents]}"


def spark_local_job():
    # the APIs the DSDL Spark modules use: SQL, ml.fpm (FPGrowth), mllib (ALS)
    from pyspark.ml.fpm import FPGrowth
    from pyspark.mllib.recommendation import ALS, Rating
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master("local[1]").appName("dsdl-smoke").getOrCreate()
    try:
        assert spark.range(100).filter("id % 2 = 0").count() == 50
        baskets = spark.createDataFrame([(0, ["a", "b"]), (1, ["a", "b", "c"]), (2, ["a", "c"])], ["id", "items"])
        assert FPGrowth(itemsCol="items", minSupport=0.5).fit(baskets).freqItemsets.count() > 0
        ratings = spark.sparkContext.parallelize([Rating(0, 0, 4.0), Rating(0, 1, 2.0), Rating(1, 1, 3.0)])
        assert ALS.train(ratings, rank=2, iterations=2).predict(1, 0) is not None
        return f"spark {spark.version} (SQL, FPGrowth, ALS), java {os.environ.get('JAVA_HOME')}"
    finally:
        spark.stop()


def gpu_build():
    import tensorflow as tf
    import torch
    assert torch.version.cuda and torch.version.cuda.startswith("12."), torch.version.cuda
    assert tf.test.is_built_with_cuda()
    return f"torch cuda {torch.version.cuda}, tf cuda {tf.sysconfig.get_build_info().get('cuda_version')}, " \
           f"gpu visible: {torch.cuda.is_available()}"


def timesfm_torch():
    import timesfm
    from timesfm.timesfm_torch import TimesFmTorch  # noqa: F401
    return f"timesfm {getattr(timesfm, '__version__', 'vendored 1.3.0.post1')}"


def rapids_import():
    try:
        import cudf
        return f"cudf {cudf.__version__} imported (GPU present)"
    except Exception as error:  # noqa: BLE001 - without a GPU only a CUDA driver error is acceptable
        message = str(error).lower()
        if any(word in message for word in ("cuda", "driver", "gpu", "libcuda", "nvml")):
            return f"needs a GPU as expected ({type(error).__name__})"
        raise


def agentic_app():
    sys.path.insert(0, "/srv")
    import importlib
    importlib.import_module("app.main_agentic")
    return "app.main_agentic imports (lifespan not started)"


CHECKS = {
    "cpu": [torch_compute, tensorflow_compute, sklearn_fit, escu_torch_models, escu_dga_savedmodel, spacy_model],
    "spark": [sklearn_fit, spark_local_job],
    "llm": [torch_compute, tensorflow_compute, sklearn_fit, agentic_app],
    "agentic": [agentic_app],
    "gpu": [torch_compute, tensorflow_compute, gpu_build, timesfm_torch, spacy_model],
    "rapids": [torch_compute, sklearn_fit, rapids_import],
}
if target in ("llm", "agentic"):
    import shutil
    shutil.copytree("/dltk/app", "/srv/app", dirs_exist_ok=True)
for func in CHECKS.get(target, []):
    check(func.__name__, func)
sys.exit(1 if failures else 0)
