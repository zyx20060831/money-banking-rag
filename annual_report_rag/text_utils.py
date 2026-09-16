from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable


_SECTION_PATTERNS = (
    (1, re.compile(r"^第[一二三四五六七八九十百0-9]+节\s*\S+")),
    (2, re.compile(r"^[一二三四五六七八九十百]+、\s*\S+")),
    (3, re.compile(r"^[（(][一二三四五六七八九十百]+[）)]\s*\S+")),
    (4, re.compile(r"^[（(]\d+[）)]\s*\S+")),
    (4, re.compile(r"^\d+(?:\.\d+){1,3}\s+\S+")),
)

_PAGE_NUMBER_PATTERNS = (
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*第?\s*\d+\s*页(?:\s*共\s*\d+\s*页)?\s*$"),
    re.compile(r"^\s*-?\s*\d+\s*-?\s*$"),
)

_QUERY_EXPANSION_RULES: tuple[
    tuple[tuple[str, ...], tuple[str, ...]], ...
] = (
    (
        ("现金分红", "留存收益", "资本需求", "再投资"),
        ("利润分配", "资本补充", "发展资金", "股东回报"),
    ),
    (
        ("区域市场", "区域布局", "渠道扩张", "跨区域", "省外"),
        ("机构网络", "营业网点", "服务网点", "渠道融合", "分支机构", "异地机构"),
    ),
    (
        ("科技赋能", "数字化", "智能化", "技术应用"),
        ("数字化转型", "金融科技", "信息科技"),
    ),
    (
        ("同行业", "细分行业", "竞争地位", "差异化优势"),
        ("行业地位", "市场份额", "竞争优势", "行业排名"),
    ),
    (
        ("经营目标", "未来一年"),
        ("经营计划", "年度目标", "经营展望", "展望", "主要任务"),
    ),
    (
        ("主要业务", "业务板块", "收入占比"),
        ("分部报告", "营业收入构成", "利息净收入", "非利息收入"),
    ),
    (
        ("研发投入", "研发支出", "研发费用"),
        ("科技投入", "信息科技投入", "金融科技", "技术研发", "研究开发费用"),
    ),
    (
        ("财务状况", "资产负债率", "现金流", "盈利能力"),
        ("偿债能力", "资本充足率", "流动性", "资产质量", "净息差"),
    ),
    (
        ("海外市场", "国际化", "境外业务"),
        (
            "境外收入",
            "海外收入",
            "境外机构",
            "分部信息",
            "地理信息",
            "地区分部",
            "境外及其他",
            "中国大陆以外",
            "国际市场",
        ),
    ),
    (
        ("外部资源", "供应方", "服务商", "资源依赖", "运营依赖"),
        ("外包风险", "业务连续性"),
    ),
    (
        ("未来战略", "发展战略", "经营重点", "可持续性"),
        (
            "发展规划",
            "战略规划",
            "经营计划",
            "未来发展",
            "发展目标",
            "战略执行",
            "重点工作",
            "展望",
            "十五五",
        ),
    ),
    (
        ("过去一年", "经营得怎么样", "重要进展", "经营情况"),
        ("经营综述", "年度经营成果", "主要经营情况", "营业收入", "净利润"),
    ),
    (
        ("重大投入", "资本开支", "资本性支出", "在建工程", "固定资产"),
        ("购建固定资产", "长期资产", "技术系统建设", "重大投资", "金融科技投入"),
    ),
    (
        ("货币政策", "利率环境", "政策利率", "利率下行", "LPR", "降息"),
        (
            "贷款市场报价利率",
            "市场利率",
            "存款利率",
            "贷款利率",
            "生息资产收益率",
            "计息负债成本率",
            "净息差",
            "利率市场化",
        ),
    ),
    (
        ("流动性", "流动性风险", "流动性管理", "存贷比"),
        ("流动性覆盖率", "流动性比例", "优质流动性资产", "流动性风险管理", "净稳定资金比例"),
    ),
    (
        ("资本充足率", "资本补充", "资本充足", "资本管理"),
        ("核心一级资本充足率", "一级资本充足率", "二级资本", "核心一级资本", "资本补充"),
    ),
    (
        ("不良贷款", "信用风险", "资产质量", "拨备", "逾期"),
        ("不良贷款率", "拨备覆盖率", "拨贷比", "逾期贷款", "关注类贷款", "减值准备", "预期信用损失"),
    ),
    (
        ("存款", "存贷比"),
        ("存款总额", "公司存款", "个人存款", "吸收存款", "客户存款"),
    ),
    (
        ("贷款", "信贷", "授信"),
        ("贷款总额", "公司贷款", "个人贷款", "发放贷款", "信贷投放", "生息资产"),
    ),
    (
        ("净息差", "净利差", "息差"),
        ("利息净收入", "生息资产", "计息负债", "存贷款利差"),
    ),
)

_QUERY_NEGATIVE_CONTEXT_RULES: tuple[
    tuple[tuple[str, ...], tuple[str, ...]], ...
] = (
    (
        ("经营目标", "未来一年"),
        ("借款人经营计划", "预期信用损失", "重要会计政策"),
    ),
    (
        ("海外市场", "国际化", "境外业务"),
        ("战略委员会主要职责", "薪酬政策", "董事及高级管理人员"),
    ),
    (
        ("未来战略", "发展战略", "经营重点", "可持续性"),
        ("战略委员会主要职责", "董事会的职责", "审计报告"),
    ),
    (
        ("重大投入", "资本开支", "资本性支出"),
        ("重要会计政策及会计估计", "在建工程成本按", "固定资产是指"),
    ),
)


def clean_line(line: str) -> str:
    line = line.replace("\x00", "").replace("　", " ")
    line = line.replace("", "[已选]")
    line = re.sub(r"[-]", "", line)
    return re.sub(r"[ \t]+", " ", line).strip()


def margin_key(line: str) -> str:
    value = unicodedata.normalize("NFKC", clean_line(line)).lower()
    value = re.sub(r"\d+", "<n>", value)
    return re.sub(r"\s+", "", value)


def find_repeated_margin_keys(
    pages: list[list[str]], margin_lines: int = 2, min_ratio: float = 0.20
) -> set[str]:
    if not pages:
        return set()

    top_counts: Counter[str] = Counter()
    bottom_counts: Counter[str] = Counter()
    for lines in pages:
        nonempty = [clean_line(line) for line in lines if clean_line(line)]
        top_counts.update({margin_key(line) for line in nonempty[:margin_lines]})
        bottom_counts.update({margin_key(line) for line in nonempty[-margin_lines:]})

    threshold = max(3, round(len(pages) * min_ratio))
    repeated = {key for key, count in top_counts.items() if key and count >= threshold}
    repeated.update(
        key for key, count in bottom_counts.items() if key and count >= threshold
    )
    return repeated


def clean_page_lines(lines: Iterable[str], repeated_margin_keys: set[str]) -> list[str]:
    cleaned: list[str] = []
    for raw_line in lines:
        line = clean_line(raw_line)
        if not line:
            continue
        if any(pattern.fullmatch(line) for pattern in _PAGE_NUMBER_PATTERNS):
            continue
        if margin_key(line) in repeated_margin_keys:
            continue
        cleaned.append(line)
    return cleaned


def detect_section(line: str) -> str | None:
    value = clean_line(line)
    if not value or len(value) > 60:
        return None
    digit_ratio = sum(char.isdigit() for char in value) / max(1, len(value))
    if digit_ratio > 0.35:
        return None
    return value if section_level(value) is not None else None


def section_level(line: str) -> int | None:
    value = clean_line(line)
    if not value or len(value) > 60:
        return None
    digit_ratio = sum(char.isdigit() for char in value) / max(1, len(value))
    if digit_ratio > 0.35:
        return None
    for level, pattern in _SECTION_PATTERNS:
        if pattern.match(value):
            return level
    return None


def normalize_for_search(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", "", value)


def query_expansion_phrases(query: str) -> list[str]:
    normalized_query = normalize_for_search(query)
    expansions: list[str] = []
    seen: set[str] = set()

    for triggers, candidates in _QUERY_EXPANSION_RULES:
        if not any(normalize_for_search(trigger) in normalized_query for trigger in triggers):
            continue

        for phrase in candidates:
            normalized_phrase = normalize_for_search(phrase)
            if normalized_phrase in normalized_query or normalized_phrase in seen:
                continue
            seen.add(normalized_phrase)
            expansions.append(phrase)

    return expansions


def query_negative_context_phrases(query: str) -> list[str]:
    normalized_query = normalize_for_search(query)
    phrases: list[str] = []
    seen: set[str] = set()
    for triggers, candidates in _QUERY_NEGATIVE_CONTEXT_RULES:
        if not any(normalize_for_search(trigger) in normalized_query for trigger in triggers):
            continue
        for phrase in candidates:
            normalized_phrase = normalize_for_search(phrase)
            if normalized_phrase in seen:
                continue
            seen.add(normalized_phrase)
            phrases.append(phrase)
    return phrases


def tokenize(text: str) -> list[str]:
    value = unicodedata.normalize("NFKC", text).lower()
    parts = re.findall(r"[a-z0-9]+(?:[._%+-][a-z0-9]+)*|[一-鿿]+", value)
    tokens: list[str] = []
    for part in parts:
        if re.fullmatch(r"[一-鿿]+", part):
            if len(part) == 1:
                tokens.append(part)
            else:
                tokens.extend(part[index : index + 2] for index in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


def compact_excerpt(text: str, limit: int = 240) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"
