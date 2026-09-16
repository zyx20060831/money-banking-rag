from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

from annual_report_rag.llm import LLMConfig
from annual_report_rag.retriever import DEFAULT_TOP_K, load_manifest
from annual_report_rag.service import AnnualReportQA, QAResult
from annual_report_rag.text_utils import compact_excerpt


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"


st.set_page_config(
    page_title="货币银行学 · 银行经营与货币政策传导问答",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def manifest_signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def friendly_error_message(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "索引 JSON 已损坏，请重新运行 build。"
    if isinstance(exc, FileNotFoundError):
        return "索引文件不完整，请重新运行 build。"
    if isinstance(exc, RuntimeError):
        return "模型接口调用失败，请检查网络、模型名称和 RAG_* 环境变量。"
    if isinstance(exc, OSError):
        return "读取本地索引失败，请检查文件权限后重新建立索引。"
    return "输入或索引配置不正确，请检查页面设置后重试。"


@st.cache_data(show_spinner=False)
def cached_manifest(data_dir_text: str, signature: str) -> dict[str, Any]:
    del signature
    return load_manifest(Path(data_dir_text))


@st.cache_resource(show_spinner=False)
def cached_service(
    data_dir_text: str,
    document_id: str,
    signature: str,
) -> AnnualReportQA:
    del signature
    return AnnualReportQA(Path(data_dir_text), document_id)


def render_document_metrics(document: dict[str, Any]) -> None:
    page_count = int(document.get("page_count", 0))
    chunk_count = int(document.get("chunk_count", 0))
    coverage = document.get("text_layer_coverage")
    coverage_text = f"{float(coverage):.1%}" if coverage is not None else "未统计"

    first, second, third = st.columns(3)
    first.metric("PDF 页数", page_count)
    second.metric("可检索文本块", chunk_count)
    third.metric("文字层覆盖率", coverage_text)


def render_evidence(result: QAResult) -> None:
    st.subheader("参考原文")
    st.caption("页码指 PDF 文件的物理页码；展开条目可查看检索片段和 BM25 分数。")

    if not result.evidence:
        st.warning("没有检索到相关原文。建议换一个更具体的问题或检查索引内容。")
        return

    for index, evidence in enumerate(result.evidence, start=1):
        chunk = evidence.chunk
        label = f"[E{index}] PDF 第 {chunk.page} 页｜{chunk.section}"
        with st.expander(label, expanded=index == 1):
            st.caption(f"BM25 相关性分数：{evidence.score:.3f}")
            st.text(compact_excerpt(chunk.text, limit=1200))


def render_result(result: QAResult) -> None:
    st.divider()
    st.subheader("回答")
    st.markdown(f"**问题：** {result.question}")
    st.caption(f"回答依据：{result.document_name}")
    st.markdown(result.answer, unsafe_allow_html=False)
    render_evidence(result)


def clear_result_when_context_changes(document_id: str, top_k: int) -> None:
    previous_document = st.session_state.get("result_document_id")
    previous_top_k = st.session_state.get("result_top_k")
    document_changed = previous_document is not None and previous_document != document_id
    top_k_changed = previous_top_k is not None and previous_top_k != top_k
    if document_changed or top_k_changed:
        st.session_state.pop("qa_result", None)
        st.session_state.pop("result_document_id", None)
        st.session_state.pop("result_top_k", None)


def main() -> None:
    st.title("🏦 货币银行学 · 经营与货币政策问答")
    st.write(
        "选择一份银行年报或宏观政策报告并输入问题。系统先检索相关段落，再让模型仅依据"
        "原文作答，同时展示参考片段和来源页码；涉及货币政策传导等宏观因果时，仅依据检索到的原文作答。"
    )

    if not MANIFEST_PATH.exists():
        st.error("尚未建立年报索引。请先在项目根目录运行下面的命令：")
        st.code("python -m annual_report_rag build --pdf-dir . --data-dir data")
        st.stop()

    try:
        signature = manifest_signature(MANIFEST_PATH)
        manifest = cached_manifest(str(DATA_DIR), signature)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[界面] 读取索引失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        st.error(friendly_error_message(exc))
        st.stop()

    documents = manifest.get("documents", [])
    if not documents:
        st.error("索引清单中没有文档。请确认项目根目录存在 PDF 后重新运行 build。")
        st.stop()

    try:
        document_by_id = {
            document["document_id"]: document for document in documents
        }
        if len(document_by_id) != len(documents):
            raise ValueError("存在重复的 document_id")
        if any(not document["source_name"] for document in document_by_id.values()):
            raise ValueError("文档名称为空")
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"索引清单中的文档信息不完整：{exc}。请重新运行 build。")
        st.stop()
    document_ids = list(document_by_id)

    with st.sidebar:
        st.header("问答设置")
        selected_id = st.selectbox(
            "选择文档",
            options=document_ids,
            format_func=lambda item: document_by_id[item]["source_name"],
        )
        top_k = st.slider(
            "参考片段数量",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
            help="数量越大，模型可参考的原文越多，但回答耗时和上下文长度也会增加。",
        )

        st.divider()
        st.subheader("模型状态")
        llm_config = LLMConfig.from_environment()
        if llm_config is None:
            st.warning("仅检索模式：未配置大模型，但仍可查看参考原文和页码。")
            st.caption("已建立索引后，本地检索不需要联网；生成综合回答需要在线或本地模型。")
        else:
            st.success("大模型已配置")
            st.caption(f"模型：{llm_config.model}")
            key_status = "已设置" if llm_config.api_key else "未设置（本地服务可为空）"
            st.caption(f"API Key：{key_status}")
            st.caption("外部模型 API 需要联网；接口指向本机模型时可完全离线运行。")

        if st.button("清除当前结果", use_container_width=True):
            st.session_state.pop("qa_result", None)
            st.session_state.pop("result_document_id", None)
            st.session_state.pop("result_top_k", None)

    clear_result_when_context_changes(selected_id, top_k)
    selected_document = document_by_id[selected_id]

    st.subheader(selected_document["source_name"])
    render_document_metrics(selected_document)
    st.caption(
        "建议问题尽量包含具体指标、年份或主题，例如：“银行的净息差、资本充足率和不良贷款率分别是多少？”"
        "或“LPR 调整对贷款利率有何影响？”"
    )

    with st.form("annual_report_question_form"):
        question = st.text_area(
            "请输入你的问题",
            placeholder="例如：报告期内银行如何响应利率下行和 LPR 调整，对净息差有何影响？",
            height=110,
            max_chars=500,
        )
        submitted = st.form_submit_button(
            "开始问答",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        cleaned_question = question.strip()
        if not cleaned_question:
            st.warning("问题不能为空，请输入一个与当前银行年报有关的问题。")
        else:
            st.session_state.pop("qa_result", None)
            st.session_state.pop("result_document_id", None)
            st.session_state.pop("result_top_k", None)
            try:
                service = cached_service(str(DATA_DIR), selected_id, signature)
                with st.spinner("正在检索年报并生成回答……"):
                    result = service.ask(
                        cleaned_question,
                        top_k=top_k,
                        llm_config=llm_config,
                    )
                st.session_state["qa_result"] = result
                st.session_state["result_document_id"] = selected_id
                st.session_state["result_top_k"] = top_k
            except (
                FileNotFoundError,
                OSError,
                ValueError,
                RuntimeError,
                json.JSONDecodeError,
            ) as exc:
                print(f"[界面] 问答失败：{type(exc).__name__}: {exc}", file=sys.stderr)
                st.error(friendly_error_message(exc))

    stored_result = st.session_state.get("qa_result")
    stored_document_id = st.session_state.get("result_document_id")
    if stored_result is not None and stored_document_id == selected_id:
        render_result(stored_result)


if __name__ == "__main__":
    main()
