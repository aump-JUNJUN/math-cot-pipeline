import difflib
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, TypeVar, Union


if TYPE_CHECKING:
    from evalscope.agent.external.runners.base import AgentRunner
    from evalscope.api.agent import AgentEnvironment, AgentStrategy, ToolHandler
    from evalscope.api.benchmark import BenchmarkMeta, DataAdapter
    from evalscope.api.evaluator import Evaluator
    from evalscope.api.filter import Filter
    from evalscope.api.metric import Aggregator, Metric
    from evalscope.api.model.model import ModelAPI
    from evalscope.api.tool import ToolInfo
    from evalscope.config import TaskConfig
    from evalscope.utils.io_utils import OutputsStructure



# TypeVar('T') 是 Python typing（类型提示）中的类型变量写法。
# 这里定义了一个类型变量 T，后续泛型类/函数（如 Registry）可以用T来表达其包含的元素类型保持一致、可变，实现类型安全的泛型注册表。
T = TypeVar('T')

class Registry(Dict[str, T]):
    """Generic name → object registry with alias support.

    Subclasses ``dict`` so existing call sites that use ``.keys()``, ``.values()``,
    ``.items()``, ``in``, ``[]`` and ``.pop()`` keep working unchanged.
    """

    def __init__(
        self, #传入的是实例
        kind: str,  #这是传入的第一个参数
        *,
        on_register: Optional[Callable[[Any, List[str]], None]] = None,
    ) -> None:
        super().__init__() #继承自字典dict  初始化一个空字典 self本身就是字典

        self.kind = kind
        self._on_register = on_register

    def register(self, name: Union[str, List[str]]) -> Callable[[T], T]:
        """Decorator that registers a value under one or more names.

        Passing a list registers the value under every name, with the first
        entry treated as the canonical / primary name.
        """
        # isinstance 是 Python 的内置类型检查函数，用于判断对象类型
        names = [name] if isinstance(name, str) else list(name)
 
        if not names:
            raise ValueError(f'{self.kind} registration requires at least one name.')

        def decorator(obj: T) -> T:
            #循环遍历names列表中的每个元素n
            for n in names:

                """
                self的含义
                调用 METRIC_REGISTRY.register("em") 时，在 register 方法里：
                self = 这个 METRIC_REGISTRY 实例
                self.kind = 'Metric'
                """

                if n in self:
                    raise ValueError(f"{self.kind} '{n}' is already registered.")
            if self._on_register is not None:
                self._on_register(obj, names)
            for n in names:
                self[n] = obj
            return obj

        return decorator  #return 可以是函数，Python 里，函数和数字、列表一样，都是对象！

    def _suggest(self, name: str, n: int = 2) -> str:
        """Return a hint string with the closest registered names by edit distance."""
        candidates = difflib.get_close_matches(name, self.keys(), n=n, cutoff=0.4)
        if candidates:
            return f"Did you mean: {', '.join(repr(c) for c in candidates)}?"
        # Fallback: show up to 10 entries if nothing is close enough
        keys = sorted(self.keys())
        if len(keys) > 10:
            return f'Available ({len(keys)} total, showing first 10): {keys[:10]}'
        return f'Available: {keys}'


    def lookup(self, name: str) -> T:
        """Get the value registered under ``name`` or raise with suggestions."""
        if name not in self:
            raise ValueError(f"{self.kind} '{name}' is not registered. {self._suggest(name)}")
        return self[name] #self是实例

"""
# BEGIN: Registry for metrics
# 这是一个 Metric 类的注册表实例，用于注册各种指标类，就是说这个是大本营呗
这一行代码所完成的内容：
属性	                           值
METRIC_REGISTRY.kind             'Metric'
METRIC_REGISTRY._on_register      None
字典内容                           {}（还没注册任何指标）

METRIC_REGISTRY: Registry[type] = Registry('Metric') #注册表的标签类型是metric是参数注册表

def register_metric(name: Union[str, List[str]]):
    # 本质上，这里是在调用 METRIC_REGISTRY 实例的 register 方法（即类中的函数），用于装饰器注册
    return METRIC_REGISTRY.register(name)  #self=METRIC_REGISTRY实例  name=acc


def get_metric(name: str) -> Type['Metric']:
    return METRIC_REGISTRY.lookup(name)
"""

# registry.py
ANSWER_METRIC_REGISTRY: Registry[type] = Registry('Answer_Metric')
COT_METRIC_REGISTRY: Registry[type] = Registry('Cot_Metric')

def register_answer_metric(name: Union[str, List[str]]) -> Callable[[Type['Metric']], Type['Metric']]:
    return ANSWER_METRIC_REGISTRY.register(name)


def register_cot_metric(name: Union[str, List[str]]) -> Callable[[Type['Metric']], Type['Metric']]:
    return COT_METRIC_REGISTRY.register(name)


def get_answer_metric(name: str) -> Type['Metric']:
    return ANSWER_METRIC_REGISTRY.lookup(name)


def get_cot_metric(name: str) -> Type['Metric']:
    return COT_METRIC_REGISTRY.lookup(name)







