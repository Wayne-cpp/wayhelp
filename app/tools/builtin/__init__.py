"""ch08 内置工具自登记(spec §1.2):装饰器挂元数据,启动扫描收集。
新工具 = 本包内新文件 + @builtin_tool,核心代码零改动。"""
import importlib
import pkgutil

from app.tools.catalog import ToolSpec

_MODULES = ("business", "faq", "ticket")


def builtin_tool(*, name, description, args_model, permission="read",
                 ownership="none", budget_priority=100):
    def deco(fn):
        fn.__tool_spec__ = ToolSpec(
            name=name, description=description,
            args_schema=args_model.model_json_schema(), args_model=args_model,
            source="builtin", permission=permission, ownership=ownership,
            budget_priority=budget_priority, fn=fn)
        return fn
    return deco


def scan_builtin_specs() -> list[ToolSpec]:
    pkg = __name__
    specs = []
    for mod_name in _MODULES:
        mod = importlib.import_module(f"{pkg}.{mod_name}")
        for attr in vars(mod).values():
            spec = getattr(attr, "__tool_spec__", None)
            if spec is not None:
                specs.append(spec)
    return specs
