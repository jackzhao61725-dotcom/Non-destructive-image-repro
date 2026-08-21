import matplotlib.image as mpimg

from scripts.render_public_readme_repeated_bec import render


def test_public_readme_repeated_bec_preview_is_renderable(tmp_path) -> None:
    output = render(tmp_path / "repeated_bec.png")

    image = mpimg.imread(output)
    assert image.ndim == 3
    assert image.shape[0] >= 900
    assert image.shape[1] >= 1100
