from pathlib import Path
from adapters.embedders import AbstractEmbeddingModel
import pytest


@pytest.mark.parametrize(
    "fixture_filepath,fixture_model",
    (
        ("test_text_filepath", "text_model",),
        ("test_image_filepath", "vision_model",),
        ("test_plot_filepath", "plot_model",),
        ("test_code_filepath", "code_model",),
    )
)
def test_clip_text_model(fixture_filepath: str, fixture_model: str, request: pytest.FixtureRequest):
    path: Path = request.getfixturevalue(fixture_filepath)
    model: AbstractEmbeddingModel = request.getfixturevalue(fixture_model)
    emb = model.embed(path.read_bytes())
    assert len(emb) == type(model).EMBEDDING_DIM
