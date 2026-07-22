import os
import numpy as np
import joblib

from art.estimators.classification.scikitlearn import (
    ScikitlearnDecisionTreeClassifier,
)
from art.attacks.evasion import DecisionTreeAttack

from get_ids import get_model

def DT_attack(surr_model_path, test_model_path, cfg):
    model = get_model("DecisionTree")
    model.load(surr_model_path)
    print("Decision Tree loaded")

    dataset_path = os.path.join(
    cfg["dir_path"],
    "..",
    "datasets",
    cfg["dataset_name"]
    )

    test_dataset_dir = os.path.join(
        dataset_path,
        "test",
        cfg["test_dataset_dir"]
    )

    model.load_dataset(
        test_dataset_dir,
        cfg["file_name"],
        "test"
    )

    X_test = np.array(model.X, dtype=np.float32)
    Y_test = np.array(model.Y)

    # ---------------- TEMPORARY DEBUG ----------------
    np.random.seed(42)

    idx = np.random.choice(len(X_test), 1000, replace=False)

    X_test = X_test[idx]
    Y_test = Y_test[idx]
    # -----------------------------------------------

    print(X_test.shape)
    print(Y_test.shape)

    classifier = ScikitlearnDecisionTreeClassifier(
        model=model.dt
    )

    print("ART wrapper created")
    
    # --------------------------------------------------
    # Create Decision Tree Attack
    # --------------------------------------------------

    offset = cfg.get("offset", 0.01)

    print("Offset =", offset)##
    print(model.dt.get_depth())##
    print(model.dt.get_n_leaves())##

    attack = DecisionTreeAttack(
        classifier=classifier,
        offset=offset
    )
    print("Decision Tree Attack initialized")
    # --------------------------------------------------
    # Generate adversarial examples
    # --------------------------------------------------

    print("Generating adversarial examples...")
    print("=" * 55)
    print("Generating Decision Tree adversarial examples...")
    print("=" * 55)
    X_adv = attack.generate(X_test)
    print("Original sample:")
    print(X_test[0])

    print("Adversarial sample:")
    print(X_adv[0])

    print("Adversarial examples generated")
    print("X_adv shape :", X_adv.shape)
    changed = np.sum(X_adv != X_test)
    print("Changed feature values:", changed)

    save_dir = os.path.join(
    dataset_path,
    "adversarial_examples",
    "DecisionTree"
    )

    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(
    os.path.join(save_dir, "DT_original_samples.csv"),
    X_test[:1000],
    delimiter=","
    )

    np.savetxt(
        os.path.join(save_dir, "DT_adversarial_samples.csv"),
        X_adv[:1000],
        delimiter=","
    )

    print("Saved first 1000 original samples")
    print("Saved first 1000 adversarial samples")
    preds = model.predict(X_adv)
    return preds, Y_test, None