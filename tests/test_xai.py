# tests/test_xai.py
def test_xai_modules():
    import src.xai.deeplift as dl, src.xai.integrated_gradients as ig
    assert callable(dl.compute_deeplift) and callable(ig.compute_ig)
