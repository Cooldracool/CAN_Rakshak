from common_imports import os
from get_ids import get_model
from evaluate import evaluation_metrics

def test_model(modelName, modelPath, cfg, adv_attack=None, image=None, TestSplit=None):
    model = get_model(modelName)

    print(f"Loading model from {os.path.normpath(modelPath)}")
    model.load(modelPath)

    dataset_path = os.path.join(
        cfg['dir_path'],
        "..",
        "datasets",
        cfg['dataset_name']
    )

    test_dataset_dir = os.path.join(
        dataset_path,
        "test",
        cfg['test_dataset_dir']
    )

    result = model.test(test_dataset_dir, cfg=cfg)

    if isinstance(result, tuple):

        if len(result) == 2:
            preds, labels = result
            evaluation_metrics(preds, labels, cfg)

        elif len(result) == 3:
            preds, labels, attack_mask = result
            evaluation_metrics(preds, labels, attack_mask, cfg)

    print("Testing Completed")