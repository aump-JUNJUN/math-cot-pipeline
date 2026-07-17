from __future__ import annotations  # 用于支持类型注解中的前向引用（Python 3.7+）

"""
参考 evalscope中GSM8K数据集数据抽取部分 -- COT和answer分割抽取
1. data中的solution列，split增加COT和answer两列  → 训练集 测试集  → 写入data/processed/train.jsonl  data/processed/test.jsonl
2. 模型评测推理预测出的结果，split增加COT和answer两列 → 写入report/*.jsonl

extract关联config中的配置文件 data.yaml  eval.yaml 使用相同策略
按照 \boxed{....} 划分 COT 和 answer
"""
"""
从 solution / generated_text 里用 \\boxed{...} 拆分 COT 和 answer。
data 与 eval 共用；config 里 strategies 目前只有 boxed。

伪：寻找maker的start_idx起始位置，找到开始为answer的索引idx
    从idx开始，找到结束为另一个花括号的索引idx end_index
    其中的就是内容:answer
    COT从索引的idx=0到start_idx，这样进行split

边界情况：没有answer 为空，或者有好几个maker 设计
"""



from dataclasses import dataclass   # 导入dataclass装饰器用于简化类的定义

_BOXED_MARKERS = ("\\boxed{", "\\fbox{")

#写成初始化类的方式调用
@dataclass(frozen=True)
class AnswerSpan:
    value: str
    start: int
    end: int
    strategy: str = "boxed"


@dataclass(frozen=True)
class ExtractResult:
    answer: str | None
    cot: str
    extract_ok: bool
    strategy: str | None = None


def _parse_braced_group(text: str, open_brace_index: int) -> tuple[str, int] | None:
    """从 '{' 起匹配成对花括号，返回 (内容, 结束位置)。"""
    depth = 1 #压栈
    idx = open_brace_index + 1
    while idx < len(text):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            # 是的，这里 depth == 0 表示找到与起始花括号 '{' 匹配的右花括号 '}'，即找到成对括号，返回内容部分和结束位置。
            if depth == 0:

                # 这里 return 返回的是一个元组 (内容, 结束位置)：
                # - 内容部分是从 open_brace_index+1 到 idx（也就是去掉包裹的那对大括号）；
                # - 结束位置是 idx+1，指向匹配的右括号的下一个字符索引（即下一个待处理位置）。
                return text[open_brace_index + 1 : idx], idx + 1
     
         
        idx += 1
    return None


def extract_boxed(text: str) -> AnswerSpan | None:
    """找 \\boxed{...} 或 \\fbox{...}；多个时取最后一个。"""
    best: AnswerSpan | None = None

    for marker in _BOXED_MARKERS:
        search_from = 0
        while True:
            # 这里使用的是 Python str 对象自带的 find 方法，用于查找 marker 在 text 中的位置（从 search_from 开始）。
            # 比如 marker 是 '\\boxed{'，那么 idx 返回的是 '\' 这个字符（也就是 marker 第一个字符）在 text 中的索引
            idx = text.find(marker, search_from)  # 这里的 search_from 参数表示从 text 的哪个索引位置开始查找 marker。比如 search_from=5，则从第6个字符开始找；这样可以连续查找 text 中多个 marker 的位置。
       
     
            if idx == -1:
                break

            # 这里是 marker 的终止位置的索引，即 '{' 的下标
            open_brace_index = idx + len(marker) - 1
     
       
            parsed = _parse_braced_group(text, open_brace_index)
            if parsed is None:
                search_from = idx + len(marker)
                continue

            value, end = parsed
            candidate = AnswerSpan(
                value=value.strip(),
                start=idx,
                end=end,
                strategy="boxed",
            )
            if best is None or candidate.start >= best.start:
                best = candidate

            search_from = idx + len(marker)

    return best


def extract_answer(text: str, strategies: list[str] | None = None) -> AnswerSpan | None:
    """
    按 config 里的 strategies 抽取；目前只支持 boxed。
    strategies 来自 data.yaml / eval.yaml，例如 ["boxed"]。
    """
    if strategies is None:
        strategies = ["boxed"]

    for name in strategies:
        if name == "boxed": #为之后留着口子，不同的数据集可能有不同的方式
            span = extract_boxed(text)
            if span is not None and span.value:
                return span
        else:
            raise ValueError(f"Unsupported strategy: {name}. Only 'boxed' is supported.")

    return None


def split_solution(text: str, strategies: list[str] | None = None) -> ExtractResult:
    """
    拆成 COT + answer。
    COT = \\boxed 前面的文字；抽不到则 extract_ok=False。
    """
    span = extract_answer(text, strategies)
    if span is None:
        return ExtractResult(
            answer=None,# 无 \boxed{} → extract_ok=False → drop_no_answer 剔除
            cot=text.strip(),
            extract_ok=False,
        )

    cot = text[: span.start].rstrip()
    return ExtractResult(
        answer=span.value,
        cot=cot,
        extract_ok=True,
        strategy=span.strategy,
    )


if __name__ == "__main__":
    sample = (
        "Step 1: compute 2+2=4.\n"
        "Step 2: multiply by 3.\n"
        "Therefore the answer is \\boxed{12}"
    )
    result = split_solution(sample, ["boxed"])
    print(result)


