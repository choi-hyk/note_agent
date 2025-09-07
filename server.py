import os
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langserve import add_routes

from note_agent.profiles import (
    create_profile,
    list_profiles,
    add_examples,
    train_profile,
    complete_with_profile,
    load_profile,
)

app = FastAPI(title="Note Agent Service")


def make_note_agent():
    """Runnable 에이전트 팩토리(공용 데모).
    - core의 구성요소로 runnable 체인을 만들어 LangServe에 바로 노출하기 위한 간단 헬퍼
    """
    from note_agent.core import (
        load_example_texts,
        build_or_load_vectorstore,
        summarize_style_rules,
        estimate_target_length,
        build_completion_chain,
        expand_to_min_length,
        EXAMPLE_DIR,
        PERSIST_DIR,
        RETRIEVE_K,
        LLM_MODEL,
    )

    examples = load_example_texts(EXAMPLE_DIR)
    vs = build_or_load_vectorstore(examples, PERSIST_DIR)
    style_rules = summarize_style_rules(examples)
    length_info = estimate_target_length(examples)
    chain = build_completion_chain(style_rules, vs, length_info, retriever_k=RETRIEVE_K)

    from langchain_core.runnables import RunnableLambda

    def _post_process(result):
        if len(result.completed_text) < length_info["min_chars"]:
            expanded = expand_to_min_length(
                text=result.completed_text,
                target_min=length_info["min_chars"],
                target_max=length_info["max_chars"],
                model=LLM_MODEL,
            )
            result.completed_text = expanded
            result.change_log.additions.append(
                f"최소 분량 미달로 사후 확장 수행(→ ≥{length_info['min_chars']}자)"
            )
        return result

    return chain | RunnableLambda(_post_process)


agent = make_note_agent()
add_routes(app, agent, path="/note-agent")


class CreateProfileReq(BaseModel):
    """프로필 생성 요청

    Attributes:
        name (str): 프로필 이름
    """

    name: str


class AddExamplesReq(BaseModel):
    """프로필에 예시 텍스트 추가

    Attributes:
        texts (List[str]): 예시 텍스트 목록
    """

    texts: List[str]


class CompleteReq(BaseModel):
    """프로필을 적용하여 완성글 요청

    Attributes:
        user_draft (str): 사용자 초안
        retriever_k (int): 검색할 문서 수
    """

    user_draft: str
    retriever_k: int = 3


@app.post("/profiles", response_model=Dict[str, Any])
def api_create_profile(req: CreateProfileReq):
    """새 프로필 생성
    - 프로필 이름은 고유해야 함

    Args:
        req (CreateProfileReq): 프로필 생성 요청 스키마

    Returns:
        meta (Dict[str, Any]): 생성된 프로필 메타데이터
    """
    meta = create_profile(req.name)
    return {"profile": meta.model_dump()}


@app.get("/profiles", response_model=Dict[str, Any])
def api_list_profiles():
    """모든 프로필 메타데이터 조회

    Returns:
        metas (Dict[str, Any]): 프로필 메타데이터 리스트
    """
    metas = [m.model_dump() for m in list_profiles()]
    return {"profiles": metas}


@app.get("/profiles/{profile_id}", response_model=Dict[str, Any])
def api_get_profile(profile_id: str):
    """특정 프로필 메타데이터 조회

    Args:
    profile_id (str): 프로필 ID

    Returns:
        meta (Dict[str, Any]): 프로필 메타데이터
    """
    try:
        meta = load_profile(profile_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"profile": meta.model_dump()}


@app.post("/profiles/{profile_id}/examples", response_model=Dict[str, Any])
def api_add_examples(profile_id: str, req: AddExamplesReq):
    """프로필에 예시 텍스트 추가

    Args:
        profile_id (str): 프로필 ID

    Returns:
        meta (Dict[str, Any]): 업데이트된 프로필 메타데이터
    """
    try:
        meta = add_examples(profile_id, req.texts)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"profile": meta.model_dump()}


@app.post("/profiles/{profile_id}/train", response_model=Dict[str, Any])
def api_train_profile(profile_id: str):
    """프로필의 메타 데이터로 학습

    Args:
        profile_id (str): 프로필 ID

    Returns:
        meta (Dict[str, Any]): 업데이트된 프로필 메타데이터
    """
    try:
        meta = train_profile(profile_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"profile": meta.model_dump()}


@app.post("/profiles/{profile_id}/complete", response_model=Dict[str, Any])
def api_complete_with_profile(profile_id: str, req: CompleteReq) -> Dict[str, Any]:
    """프로필 스타일을 적용 완성글 생성

    Args:
        profile_id (str): 프로필 ID
        req (CompleteReq): 완성 요청 스키마

    Returns:
        out (Dict[str, Any]): 완성 결과
    """
    try:
        out = complete_with_profile(
            profile_id=profile_id,
            user_draft=req.user_draft,
            retriever_k=req.retriever_k,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"output": out}
