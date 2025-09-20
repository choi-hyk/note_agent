import os
import re
import glob
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .model import ProfileHeadInfo, ProfileLengthInfo

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# =========================================
# Environment & Constants
# =========================================


load_dotenv()
assert os.getenv(
    "OPENAI_API_KEY"
), "환경 변수 OPENAI_API_KEY 가 필요합니다 (.env 설정)."

PERSIST_DIR = os.getenv("PERSIST_DIR") or "./rag_store"
EXAMPLE_DIR = os.getenv("EXAMPLE_DIR") or "./examples"
RESULTS_DIR = os.getenv("RESULTS_DIR") or "./results"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
LLM_MODEL = os.getenv("LLM_MODEL") or "gpt-4o-mini"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or "900")
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or "120")
RETRIEVE_K = int(os.getenv("RETRIEVE_K") or "3")
TEMPL_COMPLETE = float(os.getenv("TEMPL_COMPLETE") or "0.2")
TEMP_SUMMARY = float(os.getenv("TEMP_SUMMARY") or "0.3")

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

_ensure_dir(PERSIST_DIR)
_ensure_dir(EXAMPLE_DIR)
_ensure_dir(RESULTS_DIR)

# =========================================
# Schemas
# =========================================

class ChangeLog(BaseModel):
    """변경 사항 로그

    Attributes:
        fixes: 맞춤법/띄어쓰기 등 교정 사항
        additions: 자연스러운 흐름을 위한 추가 문장/단락
        factual_issues: 사실관계 오류 또는 오해 소지가 있는 부분
    """

    fixes: List[str] = Field(
        default_factory=list, description="맞춤법/띄어쓰기 등 교정 사항"
    )
    additions: List[str] = Field(
        default_factory=list, description="자연스러운 흐름을 위한 추가 문장/단락"
    )
    factual_issues: List[str] = Field(
        default_factory=list, description="사실관계 오류 또는 오해 소지가 있는 부분"
    )


class CompletionOutput(BaseModel):
    """완성된 글의 구조화된 출력

    Attributes:
        toc: 생성된 목차(상위 헤딩 순서)
        completed_text: 목차 순서대로 전개된 최종 본문
        change_log: 교정/추가/사실오류 여부를 구조화된 변경 로그로 제공
    """

    toc: List[str] = Field(default_factory=list, description="H1~H4 목차")
    completed_text: str
    change_log: ChangeLog


# =========================================
# Main Functions
# =========================================
def load_example_texts(path: str = EXAMPLE_DIR) -> List[str]:
    """RAG용 예시글을 로드하는 함수

    Args:
        path (str, optional): 예시글 폴더 경로. Defaults to EXAMPLE_DIR

    Returns:
        text (List[str]): 로드된 예시글 리스트
    """
    allowed_exts = ("*.txt", "*.md")
    paths = []
    for ext in allowed_exts:
        paths.extend(glob.glob(os.path.join(path, ext)))

    if not paths:
        raise FileNotFoundError(f"예시 글을 {EXAMPLE_DIR} 폴더에 넣어주세요.")

    paths = sorted(paths)
    texts = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            t = f.read().strip()
            if t:
                texts.append(t)
    return texts


def estimate_target_length(example_texts: List[str]) -> ProfileLengthInfo:
    """예시 글을 기반으로 길이를 추정하는 함수

    Args:
        example_text (List[str]): 예시 글 리스트

    Returns:
        length_info (ProfileLengthInfo): 평균/최소/최대 길이 정보
    """
    lengths = [len(t) for t in example_texts if t.strip()]
    if not lengths:
        return {"avg_chars": 1200, "min_chars": 1000, "max_chars": 1600}
    avg = int(sum(lengths) / len(lengths))
    min_chars = max(1000, int(avg * 0.95))
    max_chars = int(avg * 1.30)
    return ProfileLengthInfo(avg_chars=avg, min_chars=min_chars, max_chars=max_chars)

HEADER_RE = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
def define_head_info(example_texts: List[str]) -> List[ProfileHeadInfo]:
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
            results.append(ProfileHeadInfo(level=f"H{level}", title=title))
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
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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
    vs.persist()
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


# 메인 프롬프트
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 사용자의 글을 **예시 스타일**로 완성하는 한국어 도우미다.\n"
            "아래 **스타일 규칙**을 반드시 적용하라:\n{style_rules}\n\n"
            "반드시 준수할 규칙:\n"
            "1) 사용자 초안이 있을 경우 **기존 초안은 절대 바꾸지 말고**, 맞춤법/띄어쓰기/문장 부호만 교정하라.\n"
            "2) 문맥이 끊기지 않도록 자연스럽게 이어서 완성하라.\n"
            "3) **사실관계 임의 추가 금지**, 한국어만 사용, 외국어/잡문자 금지.\n"
            "4) 스타일 규칙에서 분석된 어조 중 가장 유력한 어조 하나만 일관되게 적용한다.\n"
            "5) 출력은 내부적으로 **구조화 객체(스키마)**로만 생성한다. 그 외 텍스트 생성 금지.\n"
            "6) **목차(TOC)는 H1~H3 레벨만** 생성한다. 각 항목은 (level, title) 형태이며 level은 1/2/3 중 하나만 허용한다.\n"
            "7) **본문은 반드시 이 TOC 순서**에 맞춰 작성하며, Markdown 헤더 표기(#, ##, ###)로 레벨을 정확히 표시한다.\n"
            "8) 전체 길이는 예시 기준 분량을 따른다(약 {length_avg_chars}자, 허용 범위 {length_min_chars}~{length_max_chars}자). "
            "최소 {length_min_chars}자 미만이 되지 않게 충분히 서술하라.\n\n"
            "참고 문체 조각(예시 기반 검색 결과):\n{context}\n",
        ),
        (
            "human",
            "사용자 초안(한국어, 없으면 빈 문자열 허용):\n{user_input}\n\n"
            "요구사항:\n"
            "- 먼저 이 글에 적합한 **목차(TOC)**를 H1~H3만으로 생성한다. 각 항목은 level(1/2/3), title을 포함한다.\n"
            "- 이어서 **TOC 순서대로** 본문을 작성하되, Markdown 헤더 #/##/###를 사용하여 레벨을 정확히 반영한다.\n"
            "- 길이는 {length_min_chars}~{length_max_chars}자 범위로 맞추고 가능하면 {length_avg_chars}자 근처로 작성한다.\n"
            "- 마지막으로 교정/추가/사실오류 여부를 **구조화된 변경 로그**로 제공한다.\n"
            "- 스키마 외 불필요한 텍스트는 생성하지 말라.",
        ),
    ]
)

# 사후 확장용 프롬프트
expand_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 한국어 글 편집자다. 아래 글의 **내용은 절대 바꾸지 않고**, 같은 톤/스타일로 세부 설명과 예시를 보강해라"
            "길이를 {target_min}자 이상 {target_max}자 이하로 확장하라. Markdown 헤더 구조는 유지하라.",
        ),
        ("human", "원문:\n{original}\n"),
    ]
)


def expand_to_min_length(
    text: str, target_min: int, target_max: int, model: str = LLM_MODEL
) -> str:
    """글을 최소 길이까지 확장하는 함수

    Args:
        text (str): 원문
        target_min (int): 목표 최소 길이
        target_max (int): 목표 최대 길이
        model (str, optional): 사용할 LLM 모델. Defaults to LLM_MODEL.

    Returns:
        expanded_text (str): 확장된 글
    """
    editor = ChatOpenAI(model=model, temperature=0.2)
    chain = expand_prompt | editor
    res = chain.invoke(
        {"original": text, "target_min": target_min, "target_max": target_max}
    )
    return res.content.strip()


def build_completion_chain(
    style_rules: str,
    vs: Chroma,
    length_info: dict,
    retriever_k: int = RETRIEVE_K,
    model: str = LLM_MODEL,
    temp: float = TEMPL_COMPLETE,
):
    """langchain 체인을 구축하는 함수

    Args:
        style_rules (str): 스타일 규칙
        vs (Chroma): 벡터스토어
        length_info (dict): 길이 정보
        retriever_k (int, optional): 검색할 문서 개수. Defaults to RETREIEVE_K.
        model (str, optional): 사용할 LLM 모델. Defaults to LLM_MODEL.
        temp (float, optional): LLM 온도. Defaults to TEMPL_COMPLETE.

    Returns:
        chain: 구축된 langchain 체인
    """

    retriever = vs.as_retriever(search_kwargs={"k": retriever_k})

    def format_docs(docs):
        """
        검색된 문서들을 하나의 문자열로 포맷팅하는 함수
        """
        return "\n\n---\n\n".join([d.page_content for d in docs])

    llm_structured = ChatOpenAI(model=model, temperature=temp).with_structured_output(
        CompletionOutput
    )

    chain = (
        {
            "context": retriever | format_docs,
            "user_input": RunnablePassthrough(),
            "style_rules": lambda _: style_rules,
            "length_avg_chars": lambda _: length_info["avg_chars"],
            "length_min_chars": lambda _: length_info["min_chars"],
            "length_max_chars": lambda _: length_info["max_chars"],
        }
        | prompt
        | llm_structured
    )

    return chain


def save_result(
    result: CompletionOutput, results_dir: str = RESULTS_DIR
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


# =========================================
# Note Agent
# =========================================
class NoteAgent:
    """
    사용법:
        agent = NoteAgent()
        paths = agent.run("사용자 초안 텍스트")

    LangSmith:
        chain.invoke(..., config={"metadata": {...}, "tags": [...], "project_name": "MyProject"})
    """

    def __init__(
        self,
        example_dir: str = EXAMPLE_DIR,
        persist_dir: str = PERSIST_DIR,
        results_dir: str = RESULTS_DIR,
        retriever_k: int = RETRIEVE_K,
        model: str = LLM_MODEL,
        temp_complete: float = TEMPL_COMPLETE,
        temp_summary: float = TEMP_SUMMARY,
    ):
        self.example_dir = example_dir
        self.persist_dir = persist_dir
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # 1) 예시 로드
        self.examples = load_example_texts(self.example_dir)

        # 2) RAG
        self.vs = build_or_load_vectorstore(self.examples, self.persist_dir)

        # 3) 스타일 규칙
        self.style_rules = summarize_style_rules(self.examples)

        # 4) 길이 정보
        self.length_info = estimate_target_length(self.examples)

        # 5) LLM 구성
        self.model_name = model
        self.temp_complete = temp_complete
        self.temp_summary = temp_summary

        # 6) 메인 체인
        self.main_chain = build_completion_chain(
            style_rules=self.style_rules,
            vs=self.vs,
            length_info=self.length_info,
            retriever_k=retriever_k,
            model=self.model_name,
            temp=self.temp_complete,
        )

    def run(
        self,
        user_draft: str,
        *,
        save: bool = True,
        tracing_project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Args:
            user_draft: 사용자 초안 (중간까지 쓴 글도 가능)
            save: 결과 파일 저장 여부
            tracing_project: LangSmith 프로젝트명(없으면 기본)
            tags: LangSmith 태그 리스트
            metadata: LangSmith 메타데이터(예: {"topic":"factory-method"})

        Returns:
            paths: {"completed_md": ..., "change_log_json": ...}
        """
        config = {}
        if tracing_project:
            config["project_name"] = tracing_project
        if tags:
            config["tags"] = tags
        if metadata:
            config["metadata"] = metadata

        # 1차 생성 (LangSmith에 태깅)
        result: CompletionOutput = self.main_chain.invoke(user_draft, config=config)

        # 길이 부족 시 사후 확장
        if len(result.completed_text) < self.length_info["min_chars"]:
            expanded = expand_to_min_length(
                text=result.completed_text,
                target_min=self.length_info["min_chars"],
                target_max=self.length_info["max_chars"],
                model=self.model_name,
            )
            result.completed_text = expanded
            result.change_log.additions.append(
                f"최소 분량 미달로 사후 확장 수행(→ ≥{self.length_info['min_chars']}자)"
            )

        paths = {"completed_md": "", "change_log_json": ""}
        if save:
            paths = save_result(result, self.results_dir)
        return paths
