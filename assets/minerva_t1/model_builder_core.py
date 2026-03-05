"""Core builder for the T1 MJCF model variants."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import xml.etree.ElementTree as ET

from .model_builder_assembly import (
    _apply_body_subset,
    _find_body,
    _strip_unwanted_nodes,
)
from .model_builder_config import ModelConfig, BodySubset
from .model_builder_ghosts import _apply_ghosts

# Track generated directories for cleanup on exit
_generated_dirs: set[Path] = set()


def _cleanup_generated_dirs() -> None:
    """Remove all generated directories on exit."""
    for path in _generated_dirs:
        if path.exists() and path.is_dir():
            try:
                shutil.rmtree(path)
            except Exception:
                pass  # Best-effort cleanup


atexit.register(_cleanup_generated_dirs)


class T1ModelBuilder:
    """Build MJCF variants from the canonical T1 robot model."""

    def __init__(
        self,
        base_xml_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        if base_xml_path is None:
            base_xml_path = Path(__file__).resolve().parent / "t1_robot.xml"

        self.base_xml_path = Path(base_xml_path)
        if not self.base_xml_path.exists():
            raise FileNotFoundError(
                f"Base MJCF not found: {self.base_xml_path}"
            )

        if output_dir is None:
            output_dir = self.base_xml_path.parent / "generated"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _generated_dirs.add(self.output_dir)

    def build(self, config: ModelConfig, output_path: Optional[Path] = None) -> Path:
        """Build a variant MJCF file and return its path."""
        tree = ET.parse(self.base_xml_path)
        root = tree.getroot()
        joint_specs = _collect_joint_specs(root)

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("MJCF missing <worldbody> element")

        trunk = _find_body(worldbody, "Trunk")
        if trunk is None:
            raise ValueError("MJCF missing Trunk body")

        _apply_fix_base(trunk, config.fix_base)
        worldbody = _apply_body_subset(
            root, worldbody, trunk, config.body_subset
        )
        if config.body_subset in (BodySubset.FULL, BodySubset.UPPER_BODY):
            _apply_ghosts(
                worldbody, config.ghost_type, joint_specs, self.base_xml_path
            )
        _strip_unwanted_nodes(worldbody)
        _apply_physics(root, worldbody, config.enable_physics)
        if not config.enable_collisions:
            _disable_collisions(worldbody)

        if output_path is None:
            output_path = self.output_dir / _build_filename(
                self.base_xml_path, config
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        _rewrite_paths(root, self.base_xml_path, output_path)

        ET.indent(tree, space="    ")
        tree.write(output_path, encoding="utf-8", xml_declaration=False)
        return output_path


def resolve_model_path(
    model_path: str,
    model_config: Optional[Any] = None,
) -> str:
    """Resolve a model path, generating a variant if model_config is provided."""
    if model_config is None:
        return model_path

    base_path = Path(model_path)
    if not base_path.exists():
        # Handle stale absolute paths that still include assets/...
        marker = f"assets{os.sep}"
        text = str(model_path)
        if marker in text:
            suffix = text.split(marker, 1)[1]
            try:
                from minerva_bringup.paths import resolve_path as package_resolve_path
            except Exception:  # noqa: BLE001
                package_resolve_path = None
            if package_resolve_path is not None:
                candidate = package_resolve_path(Path("assets") / suffix)
                if candidate.exists():
                    base_path = Path(candidate)

    if isinstance(model_config, ModelConfig):
        config = model_config
    else:
        config = ModelConfig.from_dict(model_config)
    builder = T1ModelBuilder(base_xml_path=base_path)
    return str(builder.build(config))


def _build_filename(base_path: Path, config: ModelConfig) -> str:
    body_tag = config.body_subset.value.lower()
    ghost_tag = config.ghost_type.value.lower().replace("_and_", "_")
    phys_tag = "phys" if config.enable_physics else "nophys"
    base_tag = "fixed" if config.fix_base else "floating"
    coll_tag = "coll" if config.enable_collisions else "nocoll"
    payload = json.dumps(config.to_dict(), sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    stem = (
        f"{base_path.stem}_{body_tag}_{ghost_tag}_{phys_tag}_{base_tag}_{coll_tag}_{digest}"
    )
    return f"{stem}.xml"


def _collect_joint_specs(root: ET.Element) -> Dict[str, Dict[str, str]]:
    specs: Dict[str, Dict[str, str]] = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        if not name:
            continue
        spec: Dict[str, str] = {}
        axis = joint.get("axis")
        if axis:
            spec["axis"] = axis
        joint_range = joint.get("range")
        if joint_range:
            spec["range"] = joint_range
        limited = joint.get("limited")
        if limited:
            spec["limited"] = limited
        if spec:
            specs[name] = spec
    return specs


def _apply_fix_base(trunk: ET.Element, fix_base: bool) -> None:
    freejoints = [child for child in trunk if child.tag == "freejoint"]
    if fix_base:
        for joint in freejoints:
            trunk.remove(joint)
        return

    if freejoints:
        return

    insert_idx = 0
    for idx, child in enumerate(list(trunk)):
        if child.tag in ("inertial", "site"):
            insert_idx = idx + 1
    trunk.insert(insert_idx, ET.Element("freejoint"))


def _apply_physics(root: ET.Element, worldbody: ET.Element, enable: bool) -> None:
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        insert_idx = 0
        for idx, child in enumerate(list(root)):
            if child.tag in ("compiler", "size"):
                insert_idx = idx + 1
        root.insert(insert_idx, option)

    option.set("integrator", "Euler")
    option.set("gravity", "0 0 -9.81" if enable else "0 0 0")

    ground = worldbody.find(".//geom[@name='ground']")
    if ground is not None:
        if enable:
            ground.set("contype", "1")
            ground.set("conaffinity", "1")
        else:
            ground.set("contype", "0")
            ground.set("conaffinity", "0")


def _disable_collisions(worldbody: ET.Element) -> None:
    for geom in worldbody.iter("geom"):
        if geom.get("name") == "ground":
            continue
        geom.set("contype", "0")
        geom.set("conaffinity", "0")


def _rewrite_paths(
    root: ET.Element, base_xml_path: Path, output_path: Path
) -> None:
    base_dir = base_xml_path.parent.resolve()
    output_dir = output_path.parent.resolve()
    if base_dir == output_dir:
        return

    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir")
        if meshdir and not Path(meshdir).is_absolute():
            mesh_src = (base_dir / meshdir).resolve()
            if mesh_src.exists():
                rel_meshdir = os.path.relpath(mesh_src, output_dir)
                if not rel_meshdir.endswith("/"):
                    rel_meshdir += "/"
                compiler.set("meshdir", rel_meshdir)

    for include in root.iter("include"):
        file_attr = include.get("file")
        if not file_attr:
            continue
        if Path(file_attr).is_absolute():
            continue
        src_path = (base_dir / file_attr).resolve()
        if not src_path.exists():
            continue
        rel_path = os.path.relpath(src_path, output_dir)
        include.set("file", Path(rel_path).as_posix())
