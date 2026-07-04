from common_imports import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import attacks.attack_handler as attack_handler


def get_attack(attack_name, cfg):
    if attack_name is None:
        return None

    for attack_class in attack_handler.__all_classes__:
        if attack_class.__name__.lower() == attack_name.lower():
            print(f"  Attack handler : {attack_class.__name__}")
            return attack_class(cfg).apply()

    raise Exception(f"{attack_name} not yet implemented")
