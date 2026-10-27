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


def escu_dga_notebook():
    # run the exported cells of the shipped ESCU notebook (what DSDL executes): imports,
    # init() and apply() on two domains
    import json
    import pandas as pd
    import shutil
    shutil.copytree("/dltk/app", "/srv/app", dirs_exist_ok=True)
    notebook = json.load(open("/dltk/notebooks/pretrained_dga_model_dsdl.ipynb"))
    namespace = {}
    for cell in notebook["cells"]:
        if cell.get("metadata", {}).get("name") in ("mltkc_import", "mltkc_init", "mltkc_apply"):
            exec(compile("".join(cell["source"]), "pretrained_dga_model_dsdl", "exec"), namespace)
    df = pd.DataFrame({"domain": ["google.com", "xjwqkzmvplrtaqhb.com"]})
    model = namespace["init"](df, {})
    result = namespace["apply"](model, df, {})
    assert list(result.columns) == ["pred_dga", "pred_dga_proba"] and len(result) == 2
    return f"pred_dga {result['pred_dga'].tolist()}"


def transformers_stack():
    """What the transformers assistants need, offline: the base models (BERT, T5) are not in the
    image and load from a local directory, so tiny random ones stand in for them, with a BERT
    vocabulary and a SentencePiece model (T5Tokenizer, is_fast=True, as the summarization
    assistant loads it)."""
    import tempfile
    import sentencepiece
    import torch
    from transformers import BertConfig, BertModel, BertTokenizer, T5Config, T5ForConditionalGeneration, T5Tokenizer
    torch.manual_seed(0)
    bert = BertModel(BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                intermediate_size=64))
    t5 = T5ForConditionalGeneration(T5Config(vocab_size=100, d_model=32, d_ff=64, d_kv=16, num_layers=2, num_heads=2,
                                             decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    with tempfile.TemporaryDirectory() as tmp:
        bert.save_pretrained(tmp + "/bert")
        t5.save_pretrained(tmp + "/t5")
        with open(tmp + "/bert/vocab.txt", "w") as f:
            f.write("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "cat", "sat"]) + "\n")
        with open(tmp + "/corpus.txt", "w") as f:
            f.write("\n".join(["the cat sat on the mat", "summarize: the dog ran in the park"] * 50))
        sentencepiece.SentencePieceTrainer.train(
            input=tmp + "/corpus.txt", model_prefix=tmp + "/t5/spiece", vocab_size=30, hard_vocab_limit=False,
            pad_id=0, eos_id=1, unk_id=2, bos_id=-1, minloglevel=2)
        bert = BertModel.from_pretrained(tmp + "/bert", local_files_only=True)
        t5 = T5ForConditionalGeneration.from_pretrained(tmp + "/t5", local_files_only=True)
        bert_tokenizer = BertTokenizer.from_pretrained(tmp + "/bert", do_lower_case=True)
        t5_tokenizer = T5Tokenizer.from_pretrained(tmp + "/t5", is_fast=True)
    ids = bert_tokenizer("The cat sat", return_tensors="pt")["input_ids"]
    assert bert_tokenizer.convert_ids_to_tokens(ids[0].tolist()) == ["[CLS]", "the", "cat", "sat", "[SEP]"]
    assert bert(input_ids=ids).last_hidden_state.shape == (1, 5, 32)
    ids = t5_tokenizer("summarize: the cat sat", return_tensors="pt")["input_ids"]
    assert t5_tokenizer.decode(ids[0], skip_special_tokens=True) == "summarize: the cat sat"
    assert t5.generate(ids, max_new_tokens=4).shape[0] == 1
    import fugashi
    import ipadic
    from janome.tokenizer import Tokenizer
    from sumeval.metrics.rouge import RougeCalculator
    assert len(fugashi.GenericTagger(ipadic.MECAB_ARGS).parse("今日は良い天気です").split()) >= 4
    assert list(Tokenizer().tokenize("今日は良い天気です", wakati=True))
    assert RougeCalculator(lang="en").rouge_n(summary="the cat sat", references="the cat sat down", n=1) > 0.5
    import transformers
    return (f"transformers {transformers.__version__}: BERT and T5 with their tokenizers from a local directory, "
            "MeCab (ipadic), Janome, ROUGE")

def spacy_model():
    import spacy
    doc = spacy.load("en_core_web_sm")("Splunk is based in San Francisco.")
    return f"entities {[e.label_ for e in doc.ents]}"


def spark_local_job():
    # the APIs the DSDL Spark modules use: SQL, ml.fpm (FPGrowth), mllib (ALS). This runs with
    # --network none, where the hostname is not in /etc/hosts: apply the entrypoint's own
    # nss_wrapper fallback, as a container start would.
    import importlib.util
    spec = importlib.util.spec_from_file_location("dsdl_entrypoint", "/dltk/entrypoint.py")
    entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entrypoint)
    env = dict(os.environ, HOME="/tmp")
    entrypoint.ensure_identity(env)
    os.environ.update(env)
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


def cuda_user_space():
    # gpu and rapids share one CUDA 12.9 user space from the nvidia-*-cu12 wheels: the CUDA
    # that RAPIDS 26.10 is built with (rapidsai/base:26.10-cuda12 is CUDA 12.9.1)
    import re
    from importlib import metadata
    import torch
    names = {(d.metadata["Name"] or ""): d.version for d in metadata.distributions()}
    toolkit = {name: version for name, version in names.items()
               if name.startswith(("nvidia-cuda-", "nvidia-cublas", "nvidia-nvjitlink", "cuda-toolkit"))}
    off = {name: version for name, version in toolkit.items() if not version.startswith("12.9.")}
    assert torch.version.cuda == "12.9" and toolkit and not off, (torch.version.cuda, off)
    cu13 = [name for name in names if re.search(r"cu(da)?13", name)]
    assert not cu13 and names.get("cuda-bindings", "12.").startswith("12."), (cu13, names.get("cuda-bindings"))
    return f"torch cuda {torch.version.cuda}, {len(toolkit)} CUDA 12.9 wheels, no CUDA 13 package"


def gpu_build():
    import tensorflow as tf
    import torch
    assert tf.test.is_built_with_cuda()
    archs = torch._C._cuda_getArchFlags().split()
    # cu129 ships SASS for Turing to Blackwell only (no PTX); Maxwell, Pascal and Volta are not built
    assert archs == ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"], archs
    return f"torch cuda {torch.version.cuda} ({' '.join(archs)}), " \
           f"tf cuda {tf.sysconfig.get_build_info().get('cuda_version')}, gpu visible: {torch.cuda.is_available()}"


def xla_toolchain():
    # TensorFlow's XLA compiles with ptxas and libdevice from nvidia-cuda-nvcc-cu12
    import subprocess
    import sysconfig
    nvcc = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cuda_nvcc")
    assert os.path.isfile(os.path.join(nvcc, "nvvm", "libdevice", "libdevice.10.bc")), nvcc
    version = subprocess.run([os.path.join(nvcc, "bin", "ptxas"), "--version"], capture_output=True, text=True,
                             check=True).stdout
    assert "release 12.9" in version, version
    return f"ptxas {version.strip().splitlines()[-1]}, libdevice present"


def tensorflow_then_triton():
    # triton (bitsandbytes, torch.compile) after tensorflow crashed the process before the
    # import-order hook of finalize-venv; a subprocess turns a crash into a FAIL
    import subprocess
    code = "import tensorflow, bitsandbytes, triton; print('triton', triton.__version__)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3"))
    assert result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()[-200:]}"
    return f"tensorflow, then bitsandbytes and {result.stdout.strip().splitlines()[-1]} in one process"


def bitsandbytes_library():
    # finalize-venv must keep the CPU build and the CUDA build bitsandbytes selects from the full
    # wheel (exact version, else newest lower, else oldest higher of the same major)
    import re
    from importlib import metadata
    import torch
    from bitsandbytes import cextension, cuda_specs
    major, minor = (int(part) for part in torch.version.cuda.split("."))
    # RECORD lists the whole wheel (metadata.files() skips the files finalize-venv removed)
    record = metadata.distribution("bitsandbytes").read_text("RECORD")
    shipped = sorted((int(m[1]), int(m[2])) for m in re.finditer(r"libbitsandbytes_cuda(\d\d)(\d+)\.so,", record))
    same = [v for v in shipped if v[0] == major]
    lower = [v for v in same if v[1] <= minor]
    expected = max(lower) if lower else min(same)
    expected_name = f"libbitsandbytes_cuda{expected[0]}{expected[1]}.so"
    specs = cuda_specs.CUDASpecs(highest_compute_capability=(8, 0), cuda_version_string=f"{major}{minor}",
                                 cuda_version_tuple=(major, minor))
    path = cextension.get_cuda_bnb_library_path(specs)
    native = sorted(p.name for p in path.parent.glob("libbitsandbytes_*.so"))
    assert path.name == expected_name and native == ["libbitsandbytes_cpu.so", expected_name], (path.name, native)
    return f"torch cuda {torch.version.cuda} loads {path.name}; {len(shipped)} CUDA builds in the wheel, 1 kept"


def timesfm_torch():
    # a tiny random-weight model through the public torch path: checkpoint load and forecast
    import numpy as np
    import timesfm
    import torch
    from timesfm import pytorch_patched_decoder as ppd
    from timesfm.timesfm_torch import TimesFmTorch
    assert timesfm.TimesFm is TimesFmTorch and "jax" not in sys.modules
    # TimesFmTorch keeps the decoder's default num_kv_heads (16), so num_heads stays 16
    hparams = timesfm.TimesFmHparams(context_len=64, horizon_len=16, input_patch_len=32, output_patch_len=32,
                                     num_layers=1, num_heads=16, model_dims=64, per_core_batch_size=2)
    config = ppd.TimesFMConfig(num_layers=1, num_heads=16, hidden_size=64, intermediate_size=64, patch_len=32,
                               horizon_len=32, head_dim=4, quantiles=hparams.quantiles)
    checkpoint = "/tmp/timesfm-smoke.ckpt"
    torch.save(ppd.PatchedTimeSeriesDecoder(config).state_dict(), checkpoint)
    model = timesfm.TimesFm(hparams=hparams, checkpoint=timesfm.TimesFmCheckpoint(version="torch", path=checkpoint))
    mean, full = model.forecast([np.sin(np.arange(80) / 5.0), np.arange(50, dtype=float)], freq=[0, 1])
    assert mean.shape == (2, 16) and full.shape[:2] == (2, 16) and np.isfinite(mean).all(), (mean.shape, full.shape)
    return f"timesfm vendored 1.3.0.post1, random-weight forecast {mean.shape} on torch {torch.__version__}"


def rapids_import():
    try:
        import cudf
        # cuDF initialises CUDA lazily: the import works without a GPU, kernels need one
        return f"cudf {cudf.__version__} imports (CUDA is initialised on first use)"
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
    "cpu": [torch_compute, tensorflow_compute, sklearn_fit, escu_torch_models, escu_dga_notebook, spacy_model,
            transformers_stack, spark_local_job],
    "llm": [torch_compute, tensorflow_compute, sklearn_fit, agentic_app],
    "agentic": [agentic_app],
    "gpu": [torch_compute, tensorflow_compute, cuda_user_space, gpu_build, xla_toolchain, tensorflow_then_triton,
            bitsandbytes_library, timesfm_torch, spacy_model, transformers_stack],
    "rapids": [torch_compute, sklearn_fit, cuda_user_space, rapids_import],
}
if target not in CHECKS:
    print(f"unknown target {target} (known: {', '.join(CHECKS)})")
    sys.exit(2)
if target in ("llm", "agentic"):
    import shutil
    shutil.copytree("/dltk/app", "/srv/app", dirs_exist_ok=True)
for func in CHECKS.get(target, []):
    check(func.__name__, func)
sys.exit(1 if failures else 0)
