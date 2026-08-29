import os
import numpy as np
import joblib

from art.estimators.classification.scikitlearn import (
    ScikitlearnDecisionTreeClassifier,
)
from art.attacks.evasion import DecisionTreeAttack

from get_ids import get_model


def DT_attack(surr_model_path, test_model_path, cfg):

    print("=" * 60)
    print("Decision Tree Attack - Targeted Evasion (Malicious → Benign)")
    print("=" * 60)

    # --------------------------------------------------
    # Load the trained Decision Tree model
    # --------------------------------------------------
    model = get_model("DecisionTree")
    model.load(surr_model_path)
    print("Decision Tree loaded successfully")

    # --------------------------------------------------
    # Load test dataset
    # --------------------------------------------------
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

    print(f"Total test samples: {len(X_test)}")
    print(f"Class distribution: {np.unique(Y_test, return_counts=True)}")

    # --------------------------------------------------
    # Filter ONLY malicious samples (labels 1, 2, 3)
    # --------------------------------------------------
    attack_mask = Y_test != 0

    X_malicious = X_test[attack_mask]
    Y_malicious = Y_test[attack_mask]

    print(f"\nMalicious samples: {len(X_malicious)}")
    print(f"  DoS (1): {np.count_nonzero(Y_malicious == 1)}")
    print(f"  Fuzzy (2): {np.count_nonzero(Y_malicious == 2)}")
    print(f"  Gear Spoofing (3): {np.count_nonzero(Y_malicious == 3)}")
    print(f"Benign samples skipped: {len(X_test) - len(X_malicious)}")

    if len(X_malicious) == 0:
        print("No malicious samples found in test set!")
        return None, None, None

    # --------------------------------------------------
    # Wrap model for ART
    # --------------------------------------------------
    classifier = ScikitlearnDecisionTreeClassifier(
        model=model.dt
    )

    print("ART wrapper created")

    # --------------------------------------------------
    # Setup Decision Tree Attack parameters
    # --------------------------------------------------
    offset = cfg.get("offset", 0.01)
    print(f"Offset: {offset}")

    attack = DecisionTreeAttack(
        classifier=classifier,
        offset=offset
    )
    print("Decision Tree Attack initialized")

    # --------------------------------------------------
    # Generate adversarial examples
    # --------------------------------------------------
    print("\nGenerating adversarial examples...")
    X_adv = attack.generate(X_malicious)

    print(f"Adversarial examples generated: {X_adv.shape}")

    # --------------------------------------------------
    # Enforce IVN Constraints (Masking + Clipping)
    # --------------------------------------------------
    # Masking: CAN ID and DLC preserved
    X_adv[:, 0] = X_malicious[:, 0]  # CAN ID
    X_adv[:, 1] = X_malicious[:, 1]  # DLC

    # Clipping: Data bytes in [0, 255]
    X_adv[:, 2:] = np.clip(X_adv[:, 2:], 0, 255)
    X_adv[:, 2:] = np.round(X_adv[:, 2:])
    X_adv = X_adv.astype(np.float32)

    print(f"CAN ID changed: {np.sum(X_adv[:,0] != X_malicious[:,0])} ✅")
    print(f"DLC changed: {np.sum(X_adv[:,1] != X_malicious[:,1])} ✅")

    # Perturbation statistics
    changed = np.sum(X_adv != X_malicious)
    print(f"Changed feature values: {changed}")

    # --------------------------------------------------
    # Predict on adversarial samples
    # --------------------------------------------------
    print("\nPredicting on adversarial samples...")
    preds_adv = model.predict(X_adv)

    # Compute ASR
    total = len(X_adv)
    evaded = np.count_nonzero(preds_adv == 0)
    print(f"\nASR: {evaded}/{total} = {evaded/total:.4%}")

    # --------------------------------------------------
    # Combine benign + malicious predictions
    # --------------------------------------------------
    X_benign = X_test[~attack_mask]
    Y_benign = Y_test[~attack_mask]

    if len(X_benign) > 0:
        preds_benign = model.predict(X_benign)
        all_preds = np.concatenate([preds_benign, preds_adv])
        all_labels = np.concatenate([Y_benign, Y_malicious])

        # Create new mask matching concatenated order
        new_mask = np.concatenate([
            np.zeros(len(preds_benign), dtype=bool),
            np.ones(len(preds_adv), dtype=bool)
        ])
    else:
        all_preds = preds_adv
        all_labels = Y_malicious
        new_mask = np.ones(len(preds_adv), dtype=bool)

    print(f"Total samples evaluated: {len(all_preds)}")
    print(f"  Benign: {len(preds_benign) if len(X_benign) > 0 else 0}")
    print(f"  Malicious: {len(preds_adv)}")

    # Return all predictions, all labels, and attack mask
    return all_preds, all_labels, new_mask