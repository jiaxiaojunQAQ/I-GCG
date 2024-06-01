# Improved Techniques for Optimization-Based Jailbreaking on Large Language Models

## Quick Start 
### 1. Generate suffix initialization
```python
python attack_llm_core_best_update_our_target.py ----behaviors_config=behaviors_ours_config.json
```

### 2. Generate new json with the initialization
```python
python generate_our_config.py
```

### 3. Conduct jailbreaking attack
```python
python run_multiple_attack_our_target.py ----behaviors_config==behaviors_ours_config_init.json
```
