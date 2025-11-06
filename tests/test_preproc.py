# tests/test_preproc.py
def test_imports():
    import src.data.preprocessing as p
    assert hasattr(p, "basic_preprocess")
