import os
import numpy as np
import tensorflow as tf

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import TensorFlowV2Classifier

from get_ids import get_model


def PGD_attack(surr_model_path, test_model_path, cfg):

    print("=" * 60)
    print("MLP PGD Attack - Targeted Evasion (Malicious → Benign)")
    print("=" * 60)

    model = get_model(cfg["model"])
    model.load(surr_model_path)
    network = model.get_network()
    print("Model loaded successfully")

    dataset_path = os.path.join(cfg["dir_path"], "..", "datasets", cfg["dataset_name"])
    test_dataset_dir = os.path.join(dataset_path, "test", cfg["test_dataset_dir"])

    model.load_dataset(test_dataset_dir, cfg["file_name"], "test")

    X_test = np.array(model.X).astype(np.float32)
    Y_test = np.array(model.Y).astype(np.int32)

    print(f"Total test samples: {len(X_test)}")

    # Filter malicious samples
    attack_mask = Y_test != 0
    X_attack = X_test[attack_mask]
    Y_attack = Y_test[attack_mask]

    print(f"Malicious samples: {len(X_attack)}")
    print(f"Benign samples skipped: {len(X_test) - len(X_attack)}")

    if len(X_attack) == 0:
        print("No malicious samples found!")
        return None, None, None

    # Target: Benign class (0)
    targets = np.zeros((len(Y_attack), 4), dtype=np.float32)
    targets[:, 0] = 1.0

    # Fix 3a: Correct loss function
    loss_object = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.NONE
    )

    def loss_fn(y_true, y_pred):
        return loss_object(y_true, y_pred)

    classifier = TensorFlowV2Classifier(
        model=network,
        nb_classes=4,
        input_shape=(10,),
        clip_values=(0.0, 255.0),
        loss_object=loss_fn,
        channels_first=False,
        preprocessing=(0, 1),
    )

    epsilon = cfg.get("epsilon", 1)
    eps_step = cfg.get("eps_step", 0.2)
    max_iter = cfg.get("max_iter", 40)

    print(f"Epsilon: {epsilon}, Step: {eps_step}, Iterations: {max_iter}")

    attack = ProjectedGradientDescent(
        estimator=classifier,
        norm=2,
        eps=epsilon,
        eps_step=eps_step,
        max_iter=max_iter,
        targeted=True,
        batch_size=8192,
    )

    print("Generating adversarial examples...")
    X_adv = attack.generate(x=X_attack, y=targets)

    # IVN Constraints
    X_adv[:, 0] = X_attack[:, 0]  # CAN ID
    X_adv[:, 1] = X_attack[:, 1]  # DLC
    X_adv[:, 2:] = np.clip(X_adv[:, 2:], 0, 255)
    X_adv[:, 2:] = np.round(X_adv[:, 2:])
    X_adv = X_adv.astype(np.float32)

    print(f"CAN ID changed: {np.sum(X_adv[:,0] != X_attack[:,0])} ✅")
    print(f"DLC changed: {np.sum(X_adv[:,1] != X_attack[:,1])} ✅")

    # Predict on adversarial samples
    preds_adv = model.predict(X_adv)

    # ASR
    total = len(X_adv)
    evaded = np.count_nonzero(preds_adv == 0)
    print(f"\nASR: {evaded}/{total} = {evaded/total:.4%}")

    # Combine benign + malicious predictions
        # Combine benign + malicious predictions
    X_benign = X_test[~attack_mask]
    Y_benign = Y_test[~attack_mask]

    if len(X_benign) > 0:
        preds_benign = model.predict(X_benign)
        all_preds = np.concatenate([preds_benign, preds_adv])
        all_labels = np.concatenate([Y_benign, Y_attack])
        
        # Create new mask matching concatenated order
        # Benign samples → False, Malicious samples → True
        new_mask = np.concatenate([
            np.zeros(len(preds_benign), dtype=bool),
            np.ones(len(preds_adv), dtype=bool)
        ])
    else:
        all_preds = preds_adv
        all_labels = Y_attack
        new_mask = np.ones(len(preds_adv), dtype=bool)

    # Return with new_mask
    return all_preds, all_labels, new_mask