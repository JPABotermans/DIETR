import pytest

def test_readme_msk():
    from dietr import DIETR
    from pathlib import Path
    conf_pth = "__config__/00-base-msk.yaml"
    file_pth = "~/data/coco/images/val2017/000000479596.jpg"
    file_pth = Path("~/data/coco/images/val2017/000000479596.jpg").expanduser()

    model = DIETR( 
        conf_pth=conf_pth, 
        )
    _ = model.predict_on_file(file_pth, plot=False)


def test_readme_box():
    from dietr import DIETR
    from pathlib import Path
    conf_pth = "__config__/00-base-box.yaml"
    file_pth = "~/data/coco/images/val2017/000000479596.jpg"
    file_pth = Path("~/data/coco/images/val2017/000000479596.jpg").expanduser()

    model = DIETR( 
        conf_pth=conf_pth, 
        )
    _ = model.predict_on_file(file_pth, plot=False)
