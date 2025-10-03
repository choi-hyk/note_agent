from operator import itemgetter
import os
import re
import json
from typing import Dict, List, Optional
from datetime import datetime

from note_agent.model import HeadInfo, ProfileLengthInfo, NoteAgentOutput, ExpandedOutput
from note_agent.config import (
    PERSIST_DIR,
    RESULTS_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    RETRIEVE_K,
    TEMP_SUMMARY,
    TEMPL_COMPLETE,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


_ensure_dir(PERSIST_DIR)
_ensure_dir(RESULTS_DIR)

#------------------------------
# Utils
#------------------------------
def estimate_target_length(example_texts: List[str]) -> ProfileLengthInfo:
    """예시 글을 기반으로 길이를 추정하는 함수

    Args:
        example_text (List[str]): 예시 글 리스트

    Returns:
        length_info (ProfileLengthInfo): 평균/최소/최대 길이 정보
    """
    lengths = [len(t) for t in example_texts if t.strip()]
    if not lengths:
        return ProfileLengthInfo(avg_chars=1200, min_chars=1000, max_chars=1600)
    avg = int(sum(lengths) / len(lengths))
    min_chars = max(1000, int(avg * 0.95))
    max_chars = int(avg * 1.30)
    return ProfileLengthInfo(avg_chars=avg, min_chars=min_chars, max_chars=max_chars)


HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def define_head_info(example_texts: List[str]) -> List[HeadInfo]:
    """예시 글을 기반으로 헤더 정보를 정의하는 함수

    Args:
        example_texts (List[str]): 예시 글 리스트

    Returns:
        results (List[ProfileHeadInfo]): 헤더 정보 리스트
    """
    results = []
    seen = set()
    for text in example_texts:
        for m in HEADER_RE.finditer(text):
            level = len(m.group(1))
            title = m.group(2).strip()
            key = (level, title)
            if key in seen:
                continue
            seen.add(key)
            results.append(HeadInfo(level=f"H{level}", title=title))
    return results


def build_or_load_vectorstore(
    example_texts: List[str], persist_dir: str = PERSIST_DIR
) -> Chroma:
    """RAG용 벡터스토어를 구축하여 Chroma를 반환하는 함수

    Args:
        example_texts (List[str]): 예시 글 리스트
        persist_dir (str, optional): 벡터스토어 영구저장 경로. Defaults to PERSIST_DIR.

    Returns:
        vs (Chroma): 구축된 Chroma 벡터스토어
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if (
        os.path.exists(persist_dir)
        and os.path.isdir(persist_dir)
        and os.listdir(persist_dir)
    ):
        return Chroma(embedding_function=embeddings, persist_directory=persist_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunks, metas = [], []

    for i, text in enumerate(example_texts, start=1):
        for ch in splitter.split_text(text):
            chunks.append(ch)
            metas.append({"source": f"example_{i}"})

    vs = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metas,
        persist_directory=persist_dir,
    )
    return vs


def summarize_style_rules(example_texts: List[str]) -> str:
    """예시 글들을 분석하여 공통 스타일 규칙을 요약하는 함수

    Args:
        example_texts (List[str]): 예시 글 리스트

    Returns:
        style_rules (str): 요약된 스타일 규칙
    """
    summarizer = ChatOpenAI(model=LLM_MODEL, temperature=TEMP_SUMMARY)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 한국어 글 스타일 분석가다. 아래 예시 글들의 **공통 스타일 규칙**만 한국어로 5~8줄 불릿으로 요약하라.\n"
                "- 어조: (~다체/ ~습니다체/ ~요체 중 무엇인지)\n"
                "- 문장 길이·호흡(짧게/보통/길게)\n"
                "- 접속어(예: 그러나/또한/즉 등) 사용 경향\n"
                "- 단락 구성(서론-본론-결론 여부, 헤더에 따른 목차 구성 여부, 예시/인용 사용 여부)\n"
                "- 어휘 톤(담백/친절/전문적 등)\n"
                "※ 한국어만 작성하고, 분석 이외의 문장은 금지한다.",
            ),
            ("human", "{examples}"),
        ]
    )
    examples_joined = "\n\n---\n\n".join(example_texts)
    chain = prompt | summarizer
    res = chain.invoke({"examples": examples_joined})
    return res.content.strip()


def save_result(
    result: NoteAgentOutput, results_dir: str = RESULTS_DIR
) -> Dict[str, str]:
    """결과를 마크다운과 JSON으로 저장하는 함수

    Args:
        result (CompletionOutput): 완성된 글 결과
        results_dir (str, optional): 결과 저장 폴더 경로. Defaults to RESULTS_DIR.

    Returns:
        paths (Dict[str, str]): 저장된 파일 경로 딕셔너리
    """
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    md_path = os.path.join(results_dir, f"completed_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result.completed_text)

    json_path = os.path.join(results_dir, f"change_log_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.change_log.model_dump(), f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료:\n- {md_path}\n- {json_path}")
    return {"completed_md": md_path, "change_log_json": json_path}


#------------------------------
# 메인 프롬프트
#------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 사용자의 글을 **예시 스타일**로 완성하는 한국어 도우미다.\n"
            "아래 **스타일 규칙**을 반드시 적용하라:\n{style_rules}\n\n"
            "반드시 준수할 규칙:\n"
            "1) 사용자 초안(user_draft)은 **의미·사실·논리 구조를 보존**하고, **맞춤법/띄어쓰기/문장부호만 교정**하여 `draft_corrected` 필드로 반환하라.\n"
            "3) 이어질 추가 본문은 `body` 필드에만 작성하라. `draft_corrected`의 문장을 수정/이동/삭제하지 말라.\n"
            "4) 문맥이 끊기지 않도록 자연스럽게 이어서 완성하라.\n"
            "5) **사실관계 임의 추가 금지**, 한국어만 사용, 외국어/잡문자 금지.\n"
            "6) 스타일 규칙에서 분석된 어조 중 가장 유력한 어조 하나만 일관되게 적용한다.\n"
            "7) 출력은 내부적으로 **구조화 객체(스키마)**로만 생성한다. 그 외 텍스트 생성 금지.\n"
            "8) **목차** {head_info}가 정의된 경우 해당 헤더를 기반으로 만든다.반드시 해당 헤더만 포함한 내용으로 만든다.\n"
             "사용자 초안의 기존 헤더 구조가 다르더라도 **반드시 {head_info} 순서로 헤더를 재구성**하라.\n"
            "9) **본문은 반드시 이 {head_info}에 맞춰 작성하며, Markdown 헤더 표기(#, ##, ###, ####)로 레벨을 정확히 표시한다.\n"
            "10) 전체 길이는 예시 기준 분량을 따른다(약 {length_avg_chars}자, 허용 범위 {length_min_chars}~{length_max_chars}자). "
            "최소 {length_min_chars}자 미만이 되지 않게 충분히 서술하라.\n\n"
            "11) 길이 가이드는 `draft_corrected + body`의 합으로 적용한다(약 {length_avg_chars}자, 허용 {length_min_chars}~{length_max_chars}자).\n\n"
            "12) 출력은 **구조화 스키마(NoteAgentOutput)**로만 생성한다. 그 외 텍스트 금지. `completed_text` 필드는 `draft_corrected` 와 `body`를 이어붙인 완성 글이다.\n\n"
            "참고 문체 조각(예시 기반 검색 결과):\n{context}\n"
        ),
        (
            "human",
            "📝 사용자 지시사항:\n{user_input}\n\n"
            "🧾 사용자 초안(없으면 빈 문자열 허용):\n{user_draft}\n\n"
            "요구사항:\n"
            "- 이어서 **{head_info} 순서대로** 본문을 작성하되, Markdown 헤더 #/##/###/####를 사용하여 레벨을 정확히 반영한다.\n"
            "- 길이는 {length_min_chars}~{length_max_chars}자 범위로 맞추고 가능하면 {length_avg_chars}자 근처로 작성한다.\n"
            "- 마지막으로 교정/추가/사실오류 여부를 **구조화된 변경 로그**로 제공한다.\n"
            "- 스키마 외 불필요한 텍스트는 생성하지 말라\n\n."
            "- **중요**\n"
            "1) 사용자 초안의 문장 내용/의미는 보존하되(맞춤법·띄어쓰기·문장부호 교정만 허용)\n"
            "2) 헤더 구조만 **{head_info}**에 맞게 재배치하라.\n"
            "3) `draft_corrected`는 오직 교정만 허용, 내용 변경 금지. 본문은 `body`에만 작성.\n"
            "4) `change_log`에 교정/추가/사실오류 여부를 구조적으로 기록."
        ),
    ]
)


#------------------------------
# 사후 확장 함수
#------------------------------
# 사후 확장용 프롬프트
expand_prompt_marker = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "입력은 두 구간으로 구성된다.\n"
            "- {draft}: **절대 수정 금지(불변)**\n"
            "- {body_text}: **이 구간만** 같은 톤으로 세부설명/예시를 추가해 보강\n\n"
            "규칙:\n"
            "- 불변 구간은 한 글자도 바꾸지 마라. (사용자 초안 텍스트는 그대로 유지)\n"
            "- 확장 구간의 Markdown 헤더 구조는 유지하되, 문단·예시·근거를 보강해 "
            "{target_min}~{target_max}자 범위로 확장하라.\n"
            "- 한국어만 사용, 사실 왜곡 금지.\n\n"
            "출력은 반드시 (ExpandedOutput)를 따라야 한다. "
            "추가 작업이 없는 필드(예: draft_corrected, head_info)는 반환하지 말고, "
            " `completed_text` 필드는 `{draft}` 와 `body`를 이어붙인 완성 글이다."
            "오직 'body', 'completed_text', 'change_log' 필드만 포함하여 응답하라."
        ),
        ("human", "{marked_input}")
    ]
)

def expand_body_with_markers(
    result: NoteAgentOutput,
    target_min: int,
    target_max: int,
    model: str,
) -> NoteAgentOutput:
    """draft는 불변, body만 확장하여 최종 텍스트 result를 반환."""
    editor = ChatOpenAI(model=model, temperature=0.2)
    structured_editor = editor.with_structured_output(ExpandedOutput)

    marked_input = (
        "\n--- [DRAFT START] ---\n"
        f"{result.draft_corrected}"
        "\n--- [DRAFT END] ---\n"
        "\n--- [BODY START] ---\n"
        f"{result.body}"
        "\n--- [BODY END] ---\n"
    )

    chain = expand_prompt_marker.partial(
        draft=result.draft_corrected, 
        body_text=result.body,
        target_min=target_min, 
        target_max=target_max
    ) | structured_editor

    expanded_output: ExpandedOutput = chain.invoke(
        {"marked_input": marked_input}
    )

    result.body = expanded_output.body
    result.completed_text = expanded_output.completed_text
    result.change_log = expanded_output.change_log 

    return result


#------------------------------
# 결과 출력 함수
#------------------------------
def finalize_with_expansion(
    result: NoteAgentOutput,
    length_info: ProfileLengthInfo,
    model: str = LLM_MODEL,
) -> NoteAgentOutput:
    """
    draft_corrected(불변) + body(확장 대상)를 결합해 completed_text를 채운다.
    길이가 부족할 때만 마커 기반으로 body만 확장.
    """
    if len(result.completed_text) >= length_info.min_chars:
        return result

    origin_length = len(result.body)
    remaining_min = max(length_info.min_chars - len(result.completed_text), 0)
    remaining_max = max(length_info.max_chars - len(result.completed_text), remaining_min + 200)

    target_min = max(remaining_min, int(0.6 * length_info.min_chars))
    target_max = max(remaining_max, target_min + 200)

    expanded_full = expand_body_with_markers(
        result=result,
        target_min=target_min,
        target_max=target_max,
        model=model,
    )

    result = expanded_full
    if hasattr(result, "change_log") and hasattr(result.change_log, "additions"):
        result.change_log.additions.append(
            f"최소 분량 미달로 본문(body)만 사후 확장(초안 불변) 기존 {origin_length}자 -> {len(result.body)}자)"
        )
    return result


#------------------------------
# 체인 함수
#------------------------------
def build_completion_chain(
    style_rules: str,
    vs: Chroma,
    length_info: ProfileLengthInfo,
    head_info: Optional[List[HeadInfo]] = None,
    retriever_k: int = RETRIEVE_K,
    model: str = LLM_MODEL,
    temp: float = TEMPL_COMPLETE,
):
    """langchain 체인을 구축하는 함수

    Args:
        style_rules (str): 스타일 규칙
        vs (Chroma): 벡터스토어
        length_info (dict): 길이 정보
        head_info (List[HeadInfo] | None): 헤더 정보
        retriever_k (int, optional): 검색할 문서 개수. Defaults to RETREIEVE_K.
        model (str, optional): 사용할 LLM 모델. Defaults to LLM_MODEL.
        temp (float, optional): LLM 온도. Defaults to TEMPL_COMPLETE.

    Returns:
        chain: 구축된 langchain 체인
    """

    retriever = vs.as_retriever(search_kwargs={"k": retriever_k})

    def _format_docs(docs):
        """
        검색된 문서들을 하나의 문자열로 포맷팅하는 함수
        """
        return "\n\n---\n\n".join([d.page_content for d in docs])

    def _rag_context(x: Dict[str, str]) -> str:
        q = (x.get("user_input") or "") + "\n\n" + (x.get("user_draft") or "")
        docs = retriever.invoke(q)
        return _format_docs(docs)

    llm_structured = ChatOpenAI(model=model, temperature=temp).with_structured_output(
        NoteAgentOutput
    )

    chain = (
        {
            "context": _rag_context,
            "user_input": itemgetter("user_input"),
            "user_draft": itemgetter("user_draft"),
            "style_rules": lambda _: style_rules,
            "head_info": lambda _: head_info,
            "length_avg_chars": lambda _: length_info.avg_chars,
            "length_min_chars": lambda _: length_info.min_chars,
            "length_max_chars": lambda _: length_info.max_chars,
        }
        | prompt
        | llm_structured
    )

    return chain
