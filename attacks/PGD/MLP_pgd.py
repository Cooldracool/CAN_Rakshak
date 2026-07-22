import os
import numpy as np
import tensorflow as tf

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import TensorFlowV2Classifier

from get_ids import get_model
from evaluate import evaluation_metrics


def PGD_attack(surr_model_path, test_model_path, cfg):

    print("=" * 60)
    print("MLP PGD Attack")
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
###
    X_test = np.array(model.X).astype(np.float32)
    Y_test = np.array(model.Y).astype(np.int32)

    # Attack only malicious samples
    attack_mask = Y_test != 0

    X_attack = X_test[attack_mask]
    Y_attack = Y_test[attack_mask]

    targets = np.zeros(len(Y_attack), dtype=np.int32)
    targets = tf.keras.utils.to_categorical(
        targets,
        num_classes=4
    )



###
    print("X_test :", X_test.shape)
    print("Y_test :", Y_test.shape)

    #iske aage will be different, for the pgd
    network = model.get_network()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False
    )

    classifier = TensorFlowV2Classifier(
        model=network,
        nb_classes=4,
        input_shape=(10,),
        loss_object=loss_object,
    )

    epsilon = cfg.get("epsilon", 1)
    eps_step = cfg.get("eps_step", 0.2)
    max_iter = cfg.get("max_iter", 40)

    print(f"Epsilon   : {epsilon}")
    print(f"Step Size : {eps_step}")
    print(f"Iterations: {max_iter}") 

    attack = ProjectedGradientDescent(
    estimator=classifier,
    norm=2,
    eps=epsilon,
    eps_step=eps_step,
    max_iter=max_iter,
    targeted=True,
    batch_size=1024
)   
    print("PGD norm : L2")
    X_adv = attack.generate(
        x=X_attack,
        y=targets
    )
    # Restore immutable CAN fields
    X_adv[:, 0] = X_attack[:, 0]
    X_adv[:, 1] = X_attack[:, 1]
    X_adv[:, 2:] = np.clip(X_adv[:, 2:], 0, 255) #clipped the value here
    X_adv[:, 2:] = np.round(X_adv[:, 2:])
    X_adv = X_adv.astype(np.float32)

    print("CAN ID changed:",
      np.sum(X_adv[:,0] != X_attack[:,0]))

    print("DLC changed:",
      np.sum(X_adv[:,1] != X_attack[:,1]))
    

    changed_payload = np.sum(
    X_adv[:,2:] != X_attack[:,2:]
    )

    print("Payload values changed:", changed_payload)

    print("Adversarial dataset generated")
    print("X_adv shape :", X_adv.shape)

    changed = np.count_nonzero(X_adv != X_attack)

    print("Changed feature values:", changed)

    perturbation = X_adv - X_attack

    print("Perturbation min:", perturbation.min())
    print("Perturbation max:", perturbation.max())

    # Copy the original test set
    X_final = X_test.copy()
    # Replace only malicious samples with adversarial ones
    X_final[attack_mask] = X_adv

    preds = model.predict(X_final)
    return preds, Y_test, None

