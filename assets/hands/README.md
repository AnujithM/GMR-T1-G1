# hands Asset Dependency Map

Hand meshes and XML fragments used by T1 MuJoCo models.

## File Structure

```text
assets/hands/
├── urdf_left_with_force_sensor/
│   ├── left_hand_with_force_sensor.xml
│   ├── meshes/*
│   └── xml/*
└── urdf_right_with_force_sensor/
    ├── right_hand_with_force_sensor.xml
    ├── meshes/*
    └── xml/*
```

## Dependency Flow

```text
assets/t1/include/t1_assets_hands.xml
  -> references meshes under assets/hands/**

assets/t1/t1_robot.xml
  -> include/t1_assets_hands.xml
  -> hand body tree links to matching hand meshes

assets/t1/model_builder.py
  -> loads t1_robot.xml and emits generated variants
```

## Notes

- Left/right hand assets are mirrored but intentionally separate.
- Mesh updates in this directory propagate to all generated model variants.
