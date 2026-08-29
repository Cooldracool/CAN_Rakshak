import os
import sys
import warnings
import yaml

# Suppress TensorFlow and absl warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Add paths BEFORE any pipeline imports
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def build_config(yaml_cfg):
    """
    Build a flat config dict directly from YAML values.
    Replaces the former populate_config_module() pattern — no intermediary
    config.py or attack_config.py modules are needed.
    All pipeline functions receive this dict as their 'cfg' parameter.
    """
    dp = yaml_cfg.get('dataset_processing', {})
    te = yaml_cfg.get('testing_and_evaluation', {})
    ap = yaml_cfg.get('adversarial_perturbation', {})
    rt = yaml_cfg.get('robust_training', {})
    

    return {
        # Base path (equivalent to DIR_PATH in former config.py — points to src/)
        'dir_path': SRC_DIR,

        # Top-level
        'dataset_name': yaml_cfg.get('dataset_name', 'CARLA'),
        'file_name':    yaml_cfg.get('file_name',    'can_data_logs.csv'),

        # dataset_processing
        'preprocess':         dp.get('preprocess',        False),
        'split':              dp.get('split',             False),
        'split_mode':         dp.get('split_mode',        'default'),
        'split_ratio':        dp.get('split_ratio',       0.2),
        'feature_extraction': dp.get('feature_extraction', True),
        'feature_extractor':  dp.get('feature_extractor', 'FrameBuilder'),

        # testing_and_evaluation
        'model':             te.get('model',             'FrameInceptionResNet'),
        'model_name':        te.get('model_name',        'model'),
        'mode':              te.get('mode',              'test'),
        'epochs':            te.get('epochs',            10),
        'train_dataset_dir': te.get('train_dataset_dir', ''),
        'test_dataset_dir':  te.get('test_dataset_dir',  ''),

        # adversarial_perturbation
        # (formerly split between config.py and attack_config.py)
        'attack_case':         ap.get('attack_case', 'mal_to_benign'),
        'adv_attack':          ap.get('adv_attack',           None),
        'attack_mode':         ap.get('attack_type',          'DoS'),   # was attack_config.ATTACK_MODE
        'adv_attack_type':     ap.get('adv_attack_type',      'blackbox'),  # was attack_config.ADV_ATTACK_TYPE
        'surrogate_model':     ap.get('surrogate_model',      ''),      # was attack_config.SURROGATE_MODEL
        'target_model':        ap.get('target_model',         ''),      # was attack_config.TARGET_MODEL
        'epsilon':             ap.get('epsilon',              1),       # was attack_config.EPSILON
        'can_id':              ap.get('id',                   '00000000000'),  # was attack_config.ID
        'eps_step':            ap.get('eps_step', 0.2),
        'max_iter':            ap.get('max_iter', 40),
        'dlc':                 ap.get('dlc',                  '1000'),  # was attack_config.DLC
        'max_injection_limit': ap.get('max_injection_limit',  15),      # was attack_config.MAX_INJECTION_LIMIT
        'offset':              ap.get('offset', 0.01),

        # robust_training
        'adv_retraining':    rt.get('adv_retraining',    False),
        'defense_method':    rt.get('defense_method',    None),
        'adv_examples_path': rt.get('adv_examples_path', None),
        'adv_samples':       rt.get('adv_samples',       800),
    }


def run_pipeline(yaml_cfg):
    """Build config and run enabled pipeline stages."""
    cfg = build_config(yaml_cfg)

    run_steps = yaml_cfg.get('run_steps', {})
    dataset_path = os.path.join(PROJECT_ROOT, "datasets", cfg['dataset_name'])

    print()
    print("=" * 55)
    print("  CAN Rakshak Pipeline")
    print("=" * 55)
    print(f"  Dataset       : {cfg['dataset_name']}")
    print(f"  File          : {cfg['file_name']}")
    print(f"  Model         : {cfg['model']}")
    print(f"  Extractor     : {cfg['feature_extractor']}")
    print("=" * 55)
    print()

    # Stage 1: Dataset Processing
    if run_steps.get('dataset_processing', False):
        print("[Stage 1/4] Dataset Processing")
        print("-" * 40)
        from preprocessing import preprocess
        from get_extractor import get_extractor
        from get_splitter import get_splitter
        preprocess(dataset_path)
        extractor = get_extractor(cfg['feature_extractor'], cfg)
        if cfg['split']:
            get_splitter(dataset_path, mode=cfg['split_mode'], feature_extractor=extractor, cfg=cfg)
        print("[Stage 1/4] Done")
        print()

    # Stage 2: Testing and Evaluation
    if run_steps.get('testing_and_evaluation', False):
        mode = cfg['mode'].lower()
        print(f"[Stage 2/4] {'Training' if mode == 'train' else 'Testing'} & Evaluation")
        print("-" * 40)
        from train import train_model
        from test import test_model
        model_name = cfg['model'] + "_" + cfg['model_name'] + ".h5"
        model_path = os.path.join(PROJECT_ROOT, "models", model_name)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if mode == 'train':
            train_model(cfg['model'], model_path, cfg)
        elif mode == 'test':
            test_model(cfg['model'], model_path, cfg, adv_attack=cfg['adv_attack'])
        else:
            raise ValueError(f"Unknown mode: {cfg['mode']}")
        print("[Stage 2/4] Done")
        print()

    # Stage 3: Adversarial Perturbation
    adv_examples_path = None
    if run_steps.get('adversarial_perturbation', False):
        print("[Stage 3/4] Adversarial Perturbation")
        print("-" * 40)
        from get_attack import get_attack
        adv_examples_path = get_attack(cfg['adv_attack'], cfg)
        print("[Stage 3/4] Done")
        print()

    # Stage 4: Robust Training
    if run_steps.get('robust_training', False):
        print("[Stage 4/4] Robust Training")
        print("-" * 40)
        from retraining import adversarial_retraining
        if adv_examples_path is None:
            adv_examples_path = yaml_cfg.get("robust_training", {}).get("adv_examples_path")

        model_name = cfg['model'] + "_" + cfg['model_name'] + ".h5"
        model_path = os.path.join(PROJECT_ROOT, "models", model_name)
        adversarial_retraining(model_path, adv_examples_path, cfg, adversarial_samples_limit=cfg['adv_samples'])
        print("[Stage 4/4] Done")
        print()

    print("=" * 55)
    print("  Pipeline Complete")
    print("=" * 55)


def main():
    yaml_path = os.path.join(SRC_DIR, "config.yaml")
    yaml_cfg = load_yaml_config(yaml_path)
    run_pipeline(yaml_cfg)


if __name__ == "__main__":
    main()

    



