# Contributing

Issues and pull requests are welcome.

```bash
pip install -e ".[dev,onnx]"
pre-commit install
make test lint
```

* Code comments and docstrings are written in English.
* Every behaviour change comes with a test; geometry and safety rules are
  checked against analytic cases or simulator ground truth.
* Keep the core free of deep-learning imports: backends are imported lazily
  inside their module.
* Run `robopercept eval --scenario urban` and `--scenario warehouse` before and
  after a change to tracking, ranging or safety, and report the numbers in the
  pull request.
