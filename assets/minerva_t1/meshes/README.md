# t1 meshes

Body mesh assets consumed by `../t1_robot.xml` and generated model variants.

## Dependency

```text
assets/t1/t1_robot.xml
  -> include/t1_assets_base.xml
  -> meshes/*.STL

assets/t1/model_builder.py
  -> builds generated XML variants that still reference meshes/*.STL
```
