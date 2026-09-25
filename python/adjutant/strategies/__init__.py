"""Builds (strategy templates). JSON builds load from `spec.build_dirs()` at import; call
`reload()` after adding files. Python templates register themselves with `@template`."""
from .base import TEMPLATES, Template, blend_goals, get, template, validate_goal, validate_template
from .spec import BuildSpec, SpecError, build_dirs, load_dirs, load_file

_PYTHON: dict[str, Template] = {}


def reload(dirs=None) -> dict[str, Template]:
    """Re-read JSON builds (Python templates stay). Returns the registry."""
    for name, t in list(TEMPLATES.items()):
        if not isinstance(t, BuildSpec):
            _PYTHON[name] = t
    TEMPLATES.clear()
    TEMPLATES.update(_PYTHON)
    TEMPLATES.update(load_dirs(build_dirs() if dirs is None else dirs))
    return TEMPLATES


reload()

__all__ = ["TEMPLATES", "Template", "BuildSpec", "SpecError", "blend_goals", "get", "template", "validate_goal",
           "validate_template", "build_dirs", "load_dirs", "load_file", "reload"]
