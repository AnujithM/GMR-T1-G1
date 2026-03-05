# right hand assets

## Structure

```text
urdf_right_with_force_sensor/
├── right_hand_with_force_sensor.xml
├── right_hand_with_force_sensor.mjcf
├── right_*.urdf
├── meshes/*
└── xml/*
```

## Dependency

```text
assets/t1/include/t1_assets_hands.xml
  -> references right hand mesh files in xml/ and meshes/
```

`right_hand_with_force_sensor.xml` is the canonical source used by model assembly.
