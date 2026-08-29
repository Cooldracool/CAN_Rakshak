# attacks/fgsm_art.py

import os
import numpy as np
import tensorflow as tf

from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod

from get_ids import get_model


def FGSM_attack(surr_model_path, test_model_path, cfg):
    """
    FGSM Attack using ART library.
    
    Follows the paper's methodology:
    - Targeted attack: Malicious → Benign (class 0)
    - Only attacks malicious samples (DoS, Fuzzy, Gear Spoofing)
    - L-infinity norm (as per paper)
    - Masking: Only D0-D7 modified, CAN ID and DLC preserved
    - Clipping: Data bytes clipped to [0, 255]
    - Epsilon values: 1 and 5 (as per paper)
    """
    
    print("=" * 60)
    print("MLP FGSM Attack using ART - Targeted Evasion (Malicious → Benign)")
    print("=" * 60)

    # --------------------------------------------------
    # Step 1: Load the trained MLP model
    # --------------------------------------------------
    model = get_model(cfg["model"])
    model.load(surr_model_path)
    network = model.get_network()
    
    print("Model loaded successfully")

    # --------------------------------------------------
    # Step 2: Load test dataset
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

    X_test = np.array(model.X).astype(np.float32)
    Y_test = np.array(model.Y).astype(np.int32)

    print(f"Total test samples: {len(X_test)}")
    print(f"Class distribution: {np.unique(Y_test, return_counts=True)}")

    # --------------------------------------------------
    # Step 3: Filter ONLY malicious samples (labels 1, 2, 3)
    # --------------------------------------------------
    malicious_mask = Y_test != 0
    
    X_malicious = X_test[malicious_mask]
    Y_malicious = Y_test[malicious_mask]
    
    print(f"\nMalicious samples: {len(X_malicious)}")
    print(f"  DoS (1): {np.count_nonzero(Y_malicious == 1)}")
    print(f"  Fuzzy (2): {np.count_nonzero(Y_malicious == 2)}")
    print(f"  Gear Spoofing (3): {np.count_nonzero(Y_malicious == 3)}")
    print(f"Benign samples skipped: {len(X_test) - len(X_malicious)}")

    if len(X_malicious) == 0:
        print("No malicious samples found in test set!")
        return None, None, None

    # --------------------------------------------------
    # Step 4: Define loss function for ART
    # --------------------------------------------------
    loss_object = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.NONE
    )

    def loss_fn(y_true, y_pred):
        return loss_object(y_true, y_pred)

    # --------------------------------------------------
    # Step 5: Wrap model for ART
    # --------------------------------------------------
    classifier = TensorFlowV2Classifier(
        model=network,
        input_shape=(10,),
        nb_classes=4,
        clip_values=(0.0, 255.0),
        loss_object=loss_fn,
        channels_first=False,
        preprocessing=(0, 1),
    )

    # --------------------------------------------------
    # Step 6: Setup FGSM attack parameters
    # --------------------------------------------------
    epsilon = cfg.get("epsilon", 5)
    
    attack = FastGradientMethod(
        estimator=classifier,
        norm=np.inf,
        eps=epsilon,
        eps_step=epsilon * 0.1,
        targeted=True,
        batch_size=8192,
        minimal=False,
    )

    print(f"\nFGSM Parameters:")
    print(f"  Epsilon: {epsilon}")
    print(f"  Norm: L-infinity (as per paper)")
    print(f"  Targeted: True (target = benign class 0)")

    # --------------------------------------------------
    # Step 7: Prepare target labels (all malicious → benign)
    # --------------------------------------------------
    y_target = np.zeros((len(X_malicious), 4), dtype=np.float32)
    y_target[:, 0] = 1.0

    # --------------------------------------------------
    # Step 8: Generate adversarial examples
    # --------------------------------------------------
    print("\nGenerating adversarial examples using ART...")
    
    try:
        X_adv = attack.generate(
            x=X_malicious,
            y=y_target,
        )
        print(f"Adversarial examples generated: {X_adv.shape}")
    except Exception as e:
        print(f"Error generating adversarial examples: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

    # --------------------------------------------------
    # Step 9: Enforce IVN Constraints (Masking + Clipping)
    # --------------------------------------------------
    X_adv[:, 0] = X_malicious[:, 0]  # CAN ID
    X_adv[:, 1] = X_malicious[:, 1]  # DLC
    X_adv[:, 2:] = np.clip(X_adv[:, 2:], 0, 255)
    X_adv[:, 0] = np.clip(X_adv[:, 0], 0, 2047)
    
    print("\nIVN Constraints Applied (Masking + Clipping):")
    print("  ✓ MASK applied: Only D0-D7 (indices 2-9) were modified")
    print("  ✓ MASK applied: CAN ID (index 0) preserved")
    print("  ✓ MASK applied: DLC (index 1) preserved")
    print("  ✓ CLIP applied: Data bytes clipped to [0, 255]")
    print("  ✓ CLIP applied: CAN ID clipped to [0, 2047]")

    # --------------------------------------------------
    # Step 10: Verify constraints
    # --------------------------------------------------
    can_id_diff = np.abs(X_adv[:, 0] - X_malicious[:, 0])
    max_can_diff = np.max(can_id_diff)
    print(f"\nVerification: Max CAN ID change: {max_can_diff:.6f} (should be 0) ✅")
    
    dlc_diff = np.abs(X_adv[:, 1] - X_malicious[:, 1])
    max_dlc_diff = np.max(dlc_diff)
    print(f"Verification: Max DLC change: {max_dlc_diff:.6f} (should be 0) ✅")
    
    min_data = np.min(X_adv[:, 2:])
    max_data = np.max(X_adv[:, 2:])
    print(f"Verification: Data bytes in range [{min_data:.2f}, {max_data:.2f}] (should be [0, 255]) ✅")

    # --------------------------------------------------
    # Step 11: Perturbation statistics
    # --------------------------------------------------
    perturbation = X_adv - X_malicious
    changed = np.count_nonzero(X_adv != X_malicious)
    print(f"Changed feature values: {changed}")
    print(f"Adversarial dataset generated")
    print(f"X_adv shape: {X_adv.shape}")
    print(f"Perturbation min: {perturbation.min():.6f}")
    print(f"Perturbation max: {perturbation.max():.6f}")
    print(f"CAN_ID perturbation: {np.unique(perturbation[:, 0])}")
    print(f"DLC perturbation: {np.unique(perturbation[:, 1])}")

    # --------------------------------------------------
    # Step 12: Predict on adversarial samples
    # --------------------------------------------------
    print("\nPredicting on adversarial samples...")
    preds = model.predict(X_adv)

    # --------------------------------------------------
    # Step 13: Compute Attack Success Rate (ASR)
    # --------------------------------------------------
    total_attacks = len(X_adv)
    successful_evasions = np.count_nonzero(preds == 0)
    asr = successful_evasions / total_attacks if total_attacks > 0 else 0.0

    print("\n" + "=" * 60)
    print("ATTACK RESULTS (Evasion: Malicious → Benign)")
    print("=" * 60)
    print(f"Total malicious samples attacked: {total_attacks}")
    print(f"Successfully evaded (predicted as benign): {successful_evasions}")
    print(f"Attack Success Rate (ASR): {asr:.4%}")
    print("=" * 60)

    # --------------------------------------------------
    # Step 14: Per-class breakdown
    # --------------------------------------------------
    print("\nPer-class breakdown (original labels):")
    for label, name in [(1, "DoS"), (2, "Fuzzy"), (3, "Gear Spoofing")]:
        class_mask = Y_malicious == label
        class_total = np.count_nonzero(class_mask)
        if class_total > 0:
            class_preds = preds[class_mask]
            evaded = np.count_nonzero(class_preds == 0)
            class_asr = evaded / class_total
            print(f"  {name}: {evaded}/{class_total} evaded (ASR: {class_asr:.4%})")

    print("\n" + "=" * 60)

    # --------------------------------------------------
    # Return predictions for ALL test samples (benign + malicious)
    # --------------------------------------------------
    print("\nGenerating predictions for ALL test samples...")

    X_benign = X_test[~malicious_mask]
    Y_benign = Y_test[~malicious_mask]

    if len(X_benign) > 0:
        preds_benign = model.predict(X_benign)
        print(f"Benign samples (not attacked): {len(preds_benign)}")
    else:
        preds_benign = np.array([])

    print(f"Malicious samples (attacked): {len(preds)}")

    if len(preds_benign) > 0:
        all_preds = np.concatenate([preds_benign, preds])
        all_labels = np.concatenate([Y_benign, Y_malicious])
    else:
        all_preds = preds
        all_labels = Y_malicious

    print(f"Total samples evaluated: {len(all_preds)}")
    print(f"  Benign: {len(preds_benign)}")
    print(f"  Malicious: {len(preds)}")

    return all_preds, all_labels, malicious_mask