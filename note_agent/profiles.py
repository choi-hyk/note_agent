import os
import json
import uuid
from typing import List, Dict, Any
from datetime import datetime

from .core import (
    build_or_load_vectorstore,
    summarize_style_rules,
    estimate_target_length,
    build_completion_chain,
    expand_to_min_length,
    LLM_MODEL,
)

from .model import ProfileMeta

BASE_DIR = os.getenv("PROFILES_DIR") or "./profiels"


def _profile_dir(profile_id: str) -> str:
    return os.path.join(BASE_DIR, profile_id)


def _examples_dir(profile_id: str) -> str:
    return os.path.join(_profile_dir(profile_id), "examples")


def _persist_dir(profile_id: str) -> str:
    return os.path.join(_profile_dir(profile_id), "rag_store")


def _meta_path(profile_id: str) -> str:
    return os.path.join(_profile_dir(profile_id), "profile.json")


def create_profile(name: str) -> ProfileMeta:
    """프로필을 생성하는 함수

    Args:
        name: 프로필 이름

    Returns:
        ProfileMeta: 생성된 프로필 메타데이터
    """
    os.makedirs(BASE_DIR, exist_ok=True)
    profile_id = str(uuid.uuid4())
    pdir = _profile_dir(profile_id)
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(_examples_dir(profile_id), exist_ok=True)

    meta = ProfileMeta(
        profile_id=profile_id,
        name=name,
        created_at=datetime.now().isoformat(),
        style_rules=None,
        length_info=None,
        persist_dir=_persist_dir(profile_id),
        examples_count=0,
    )
    with open(_meta_path(profile_id), "w", encoding="utf-8") as f:
        json.dump(meta.model_dump(), f, ensure_ascii=False, indent=2)
    return meta


def load_profile(profile_id: str) -> ProfileMeta:
    """
    프로필 id를 통해 메타데이터를 로드하는 함수

    Args:
        profile_id: 프로필 ID

    Returns:
        ProfileMeta: 프로필 메타데이터
    """
    with open(_meta_path(profile_id), "r", encoding="utf-8") as f:
        return ProfileMeta(**json.load(f))


def list_profiles() -> List[ProfileMeta]:
    """모든 프로필 메타데이터 조회

    Returns:
        metas (List[ProfileMeta]): 프로필 메타데이터 리스트
    """
    if not os.path.isdir(BASE_DIR):
        return []
    metas = []
    for pid in os.listdir(BASE_DIR):
        mpath = _meta_path(pid)
        if os.path.isfile(mpath):
            with open(mpath, "r", encoding="utf-8") as f:
                metas.append(ProfileMeta(**json.load(f)))
    return metas


def add_examples(profile_id: str, texts: List[str]) -> ProfileMeta:
    """사용자 예시글(plain text) 저장

    Args:
        profile_id: 프로필 ID
        texts: 예시 텍스트 리스트

    Returns:
        metas (ProfileMeta): 업데이트된 프로필 메타데이터
    """
    ex_dir = _examples_dir(profile_id)
    os.makedirs(ex_dir, exist_ok=True)
    existing = len([p for p in os.listdir(ex_dir) if p.endswith((".txt", ".md"))])
    for i, t in enumerate(texts, start=1):
        path = os.path.join(ex_dir, f"example_{existing + i}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(t.strip())

    meta = load_profile(profile_id)
    meta.examples_count = len(
        [p for p in os.listdir(ex_dir) if p.endswith((".txt", ".md"))]
    )
    with open(_meta_path(profile_id), "w", encoding="utf-8") as f:
        json.dump(meta.model_dump(), f, ensure_ascii=False, indent=2)
    return meta


def _load_examples_from_disk(profile_id: str) -> List[str]:
    """디스크에서 예시글을 로드하는 함수

    Args:
        profile_id: 프로필 ID

    Returns:
        texts (List[str]): 예시 텍스트 리스트
    """
    ex_dir = _examples_dir(profile_id)
    if not os.path.isdir(ex_dir):
        return []
    texts = []
    for fname in sorted(os.listdir(ex_dir)):
        if fname.endswith((".txt", ".md")):
            with open(os.path.join(ex_dir, fname), "r", encoding="utf-8") as f:
                t = f.read().strip()
                if t:
                    texts.append(t)
    return texts


def train_profile(profile_id: str) -> ProfileMeta:
    """예시글 기반으로 벡터스토어/스타일 규칙/길이 정보 생성

    Args:
        profile_id: 프로필 ID

    Returns:
        ProfileMeta: 업데이트된 프로필 메타데이터
    """
    meta = load_profile(profile_id)
    examples = _load_examples_from_disk(profile_id)
    if len(examples) < 1:
        raise ValueError("예시글이 최소 1개 이상 필요합니다. 먼저 업로드하세요.")

    os.makedirs(meta.persist_dir, exist_ok=True)
    _ = build_or_load_vectorstore(example_texts=examples, persist_dir=meta.persist_dir)

    style_rules = summarize_style_rules(examples)
    length_info = estimate_target_length(examples)

    meta.style_rules = style_rules
    meta.length_info = length_info
    with open(_meta_path(profile_id), "w", encoding="utf-8") as f:
        json.dump(meta.model_dump(), f, ensure_ascii=False, indent=2)
    return meta


def complete_with_profile(
    profile_id: str,
    user_draft: str,
    *,
    retriever_k: int = 3,
) -> Dict[str, Any]:
    """프로필 스타일로 글 완성(TOC/본문/변경로그 반환)

    Args:
        profile_id: 프로필 ID
        user_draft: 사용자가 작성한 초안
        retriever_k: RAG 검색 시 상위 k개 문서 활용(기본 3)

    Returns:
        out (Dict[str, Any]): 완성 결과
    """
    meta = load_profile(profile_id)
    if not (meta.style_rules and meta.length_info and meta.persist_dir):
        raise ValueError(
            "해당 프로필은 아직 학습되지 않았습니다. /profiles/{id}/train 을 먼저 호출하세요."
        )

    # 프로필 전용 벡터스토어 로드
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma

    embeddings = OpenAIEmbeddings()
    vs = Chroma(embedding_function=embeddings, persist_directory=meta.persist_dir)

    # 체인 구성 및 실행
    chain = build_completion_chain(
        style_rules=meta.style_rules,
        vs=vs,
        length_info=meta.length_info,
        retriever_k=retriever_k,
    )
    result = chain.invoke(user_draft)

    # 길이 보정
    if len(result.completed_text) < meta.length_info["min_chars"]:
        expanded = expand_to_min_length(
            text=result.completed_text,
            target_min=meta.length_info["min_chars"],
            target_max=meta.length_info["max_chars"],
            model=LLM_MODEL,
        )
        result.completed_text = expanded
        result.change_log.additions.append(
            f"최소 분량 미달로 사후 확장 수행(→ ≥{meta.length_info['min_chars']}자)"
        )

    return {
        "toc": result.toc,
        "completed_text": result.completed_text,
        "change_log": result.change_log.model_dump(),
        "profile_id": profile_id,
    }
