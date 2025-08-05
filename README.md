# Maps

The map files of CTC tasks and simplified variants should copy to path (~/StarCraftII/Maps/SMAC_Maps/)



# Register

Register these maps by inserting the following text into /smac/smac/env/starcraft2/maps/smac_maps.py

```python
    "Defense3Subtask": {
        "n_agents": 9,
        "n_enemies": 9,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "Defense4Subtask": {
        "n_agents": 12,
        "n_enemies": 12,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "Mixed2Subtask": {
        "n_agents": 6,
        "n_enemies": 6,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "Mixed3Subtask": {
        "n_agents": 9,
        "n_enemies": 9,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
```
