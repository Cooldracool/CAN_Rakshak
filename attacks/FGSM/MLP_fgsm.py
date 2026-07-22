import os
import numpy as np
import tensorflow as tf
import joblib

from get_ids import get_model
from evaluate import evaluation_metrics

def FGSM_attack(surr_model_path, test_model_path, cfg):

    print("=" * 60)
    print("MLP FGSM Attack")
    print("=" * 60)

    # --------------------------------------------------
    # Load trained MLP
    # --------------------------------------------------
    model = get_model(cfg["model"])
    model.load(surr_model_path)

    print("Model loaded successfully")

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

    X_test = np.array(model.X).astype(np.float32)
    Y_test = np.array(model.Y).astype(np.int32)

    print("X_test :", X_test.shape)
    print("Y_test :", Y_test.shape)

    ###############################################################
    # Step 1 : Compute FGSM Gradient
    ###############################################################

    batch_size = 8192

    all_gradients = []

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    network = model.get_network()

    for start in range(0, len(X_test), batch_size):

        end = min(start + batch_size, len(X_test))

        x = tf.convert_to_tensor(
            X_test[start:end],
            dtype=tf.float32
        )

        y = tf.convert_to_tensor(
            Y_test[start:end],
            dtype=tf.int32
        )

        with tf.GradientTape() as tape:

            tape.watch(x)

            logits = network(x, training=False)

            loss = loss_fn(y, logits)

        gradient = tape.gradient(loss, x)

        if gradient is None:
            raise RuntimeError(f"Gradient is None in batch {start}")

        all_gradients.append(gradient.numpy())


    
    
    gradient = np.vstack(all_gradients)

    epsilon = cfg.get("epsilon", 10)
    print("Epsilon =", epsilon)
    perturbation = epsilon * np.sign(gradient)
    mask = np.array(
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        dtype=np.float32
    )

    perturbation *= mask

    X_adv = X_test + perturbation
    X_adv[:, 0] = X_test[:, 0]      # CAN ID
    X_adv[:, 1] = X_test[:, 1]      # DLC

    X_adv[:, 2:] = np.clip(X_adv[:, 2:], 0, 255)

    ############################################################
    # Save first 1000 clean and adversarial samples
    ############################################################

    output_dir = os.path.join(
        cfg["dir_path"],
        "..",
        "datasets",
        cfg["dataset_name"],
        "FGSM_Output"
    )

    os.makedirs(output_dir, exist_ok=True)

    num_samples = 1000

    original_file = os.path.join(
        output_dir,
        "Original_1000.csv"
    )

    adv_file = os.path.join(
        output_dir,
        "FGSM_1000.csv"
    )

    np.savetxt(
        original_file,
        X_test[:num_samples],
        delimiter=",",
        fmt="%.6f"
    )

    np.savetxt(
        adv_file,
        X_adv[:num_samples],
        delimiter=",",
        fmt="%.6f"
    )

    print(f"Saved {num_samples} original samples")
    print(f"Saved {num_samples} adversarial samples")

    changed = np.count_nonzero(X_adv != X_test)
    print("Changed feature values:", changed)

    print("Adversarial dataset generated")
    print("X_adv shape:", X_adv.shape)
    print("Perturbation min:", perturbation.min())
    print("Perturbation max:", perturbation.max())

    print("CAN_ID perturbation:", np.unique(perturbation[:, 0]))
    print("DLC perturbation   :", np.unique(perturbation[:, 1]))

    preds = model.predict(X_adv)
    return preds, Y_test, None

    