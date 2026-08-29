from common_imports import (
    os, np, plt, datetime,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
)


def evaluation_metrics(all_preds, all_labels, cfg, attack_mask=None):
    """
    Evaluate model performance with optional adversarial attack metrics.
    
    Args:
        all_preds: Predictions from the model
        all_labels: Ground truth labels
        cfg: Configuration dictionary
        attack_mask: Boolean mask indicating which samples were attacked
                    (only needed for adversarial evaluation)
    """
    
    if all_labels is None or all_preds is None:
        print("No predictions or labels to evaluate")
        return

    print("Inside evaluation metrics")
    dir_path = cfg['dir_path']
    dataset_name = cfg['dataset_name']
    adv_attack = cfg.get('adv_attack', False)
    adv_attack_type = cfg.get('adv_attack_type', '')
    model = cfg.get('model', '')
    model_name = cfg.get('model_name', '')

    dataset_path = os.path.join(dir_path, "..", "datasets", dataset_name)
    
    if adv_attack:
        result_path = os.path.join(dataset_path, "Results", adv_attack, adv_attack_type)
    else:
        result_path = os.path.join(dataset_path, "Results", model + "_" + model_name)
    
    os.makedirs(result_path, exist_ok=True)

    timestamp = datetime.now().strftime("_%Y_%m_%d_%H%M%S")

    if adv_attack:
        filename = f"{adv_attack_type}_dos_{timestamp}.png"
    else:
        filename = f"{model_name}_{timestamp}.png"

    print("Number of predictions:", len(all_preds))
    print("Unique predictions:", np.unique(all_preds, return_counts=True))
    print("Unique labels:", np.unique(all_labels, return_counts=True))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    print("Confusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["B", "D", "F", "G"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')

    output_path = os.path.join(result_path, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

    # Performance metrics
    IDS_accu = accuracy_score(all_labels, all_preds)
    IDS_prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    IDS_recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    IDS_F1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print("----------------IDS Performance Metric----------------")
    print(f'Accuracy: {IDS_accu:.4f}')
    print(f'Precision: {IDS_prec:.4f}')
    print(f'Recall: {IDS_recall:.4f}')
    print(f'F1 Score: {IDS_F1:.4f}')

    # Adversarial metrics (if attack was performed)
    if adv_attack and attack_mask is not None:
        # Only consider malicious samples that were attacked
        malicious_preds = all_preds[attack_mask]
        malicious_labels = all_labels[attack_mask]
        
        # Success = malicious predicted as benign (class 0)
        successful_evasions = np.count_nonzero(malicious_preds == 0)
        total_attacks = np.count_nonzero(attack_mask)
        
        asr = successful_evasions / total_attacks if total_attacks > 0 else 0.0
        
        # Per-class breakdown
        print("\n----------------Adversarial Attack Performance Metric----------------")
        print(f"Total attacked packets: {total_attacks}")
        print(f"Successful evasions: {successful_evasions}")
        print(f"Attack Success Rate (ASR): {asr:.4%}")
        
        print("\nPer-class evasion rates:")
        for label, name in [(1, "DoS"), (2, "Fuzzy"), (3, "Gear Spoofing")]:
            class_mask = malicious_labels == label
            class_total = np.count_nonzero(class_mask)
            if class_total > 0:
                class_preds = malicious_preds[class_mask]
                evaded = np.count_nonzero(class_preds == 0)
                class_asr = evaded / class_total
                print(f"  {name}: {evaded}/{class_total} evaded (ASR: {class_asr:.4%})")