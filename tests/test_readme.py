import pytest

def test_readme_msk():
    from dietr import DIETR
    from pathlib import Path
    conf_pth = "__config__/00-base-msk.yaml"
    file_pth = Path("tests/test_data/coco_mini/images/000000005802.jpg").expanduser()

    model = DIETR( 
        conf_pth=conf_pth, 
        )
    _ = model.predict_on_file(file_pth, plot=False)


def test_readme_box():
    from dietr import DIETR
    from pathlib import Path
    conf_pth = "__config__/00-base-box.yaml"
    file_pth = Path("tests/test_data/coco_mini/images/000000005802.jpg").expanduser()

    model = DIETR( 
        conf_pth=conf_pth, 
        )
    _ = model.predict_on_file(file_pth, plot=False)



def test_finetune():
    from dietr.trn import train
    train(config_pth = "tests/test_data/test_config_tune.yaml")