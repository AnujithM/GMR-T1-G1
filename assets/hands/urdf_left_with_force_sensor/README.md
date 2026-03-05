# left hand assets

## Structure

```text
urdf_left_with_force_sensor/
├── left_hand_with_force_sensor.xml
├── left_hand_with_force_sensor.mjcf
├── left_*.urdf
├── meshes/*
└── xml/*
```

## Dependency

```text
assets/t1/include/t1_assets_hands.xml
  -> references left hand mesh files in xml/ and meshes/
```

`left_hand_with_force_sensor.xml` is the canonical source used by model assembly.
