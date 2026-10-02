# Contributing to CAN Rakshak

Thanks for your interest! Ways to help:

- Add a new IDS model, attack, feature extractor, splitter or defense
- Add support for a new CAN dataset
- Report bugs or unclear documentation via GitHub Issues

## How to add a component

1. Find the base class in the *Extending the Framework* table in the README.
2. Create a new file in the matching folder and inherit from that base class.
3. Components in `attacks/attack_handler/` are auto-registered. Select yours by class name in `src/config.yaml`.
4. Run the pipeline on at least one dataset and include the results in your PR.
