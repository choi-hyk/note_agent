from typing import Dict, Any, List, Any, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from langserve import add_routes
from langchain_core.runnables import Runnable

from note_agent.profiles import (
    create_profile,
    update_profile,
    delete_profile,
    list_profiles,
    add_examples,
    train_profile,
    complete_with_profile,
    load_profile,
)

from note_agent.model import (
    CompleteReq,
    CreateProfileReq,
    NoteAgentInput,
    NoteAgentOutput,
    UpdateProfileReq
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.profile_agents = {} 
    yield
    app.state.profile_agents.clear()

app = FastAPI(
    title="Note Agent Service",
    description="프로필을 기반으로 글을 완성시켜주는 Agent.",
    debug=True,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "NoteAgent", "description": "프로필 기반 글 생성"},
        {"name": "Profiles", "description": "프로필 생성/관리"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ------------------------------
# NoteAgent
# -----------------------------
def _ensure_profile_exists(profile_id: str):
    try:
        return load_profile(profile_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="프로필을 찾을 수 없습니다.")

def _ensure_profile_agent(profile_id: str):
    cached = app.state.profile_agents.get(profile_id)
    if cached:
        return cached

    def _run(user_draft: str, retriever_k: int | None):
        return complete_with_profile(
            profile_id=profile_id,
            user_draft=user_draft,
            retriever_k=retriever_k if retriever_k is not None else 3,
            cache=cached
        )

    async def _arun(user_draft: str, retriever_k: int | None):
        return _run(user_draft, retriever_k)

    app.state.profile_agents[profile_id] = (_run, _arun)
    return app.state.profile_agents[profile_id]

class NoteAgent(Runnable):
    """
    LangServe에 등록할 Runnable 구현.
    - 입력(JSON): {"profile_id": "...", "user_draft": "...", "retriever_k": 3}
    - 출력(JSON): complete_with_profile(...) 반환 dict
    """

    def invoke(self, input: dict, config=None):
        if not isinstance(input, dict):
            raise ValueError("input은 객체(JSON)여야 합니다.")

        profile_id = input.get("profile_id")
        if not profile_id:
            raise ValueError("profile_id는 필수입니다.")

        user_draft = input.get("user_draft") or ""
        if not user_draft:
            raise ValueError("user_draft는 필수입니다.")

        retriever_k = input.get("retriever_k", 3)

        _ensure_profile_exists(profile_id)

        run, _ = _ensure_profile_agent(profile_id)
        return run(user_draft=user_draft, retriever_k=retriever_k)

    async def ainvoke(self, input: dict, config=None):
        if not isinstance(input, dict):
            raise ValueError("input은 객체(JSON)여야 합니다.")

        profile_id = input.get("profile_id")
        if not profile_id:
            raise ValueError("profile_id는 필수입니다.")

        user_draft = input.get("user_draft") or ""
        if not user_draft:
            raise ValueError("user_draft는 필수입니다.")

        retriever_k = input.get("retriever_k", 3)

        _ensure_profile_exists(profile_id)

        _, arun = _ensure_profile_agent(profile_id)
        return await arun(user_draft=user_draft, retriever_k=retriever_k)

    def stream(self, input: dict, config=None):
        yield self.invoke(input, config)

note_agent = NoteAgent()
add_routes(app, note_agent, path="/note-agent",     input_type=NoteAgentInput,
    output_type=NoteAgentOutput,)


# ---------------------------
# Profile routers
# ---------------------------
@app.post(
    "/profiles",
    tags=["Profiles"],
    summary="새 프로필 생성",
    response_model=Dict[str, Any],
    response_description="생성된 프로필 메타데이터를 반환합니다.",
)
def api_create_profile(req: CreateProfileReq):
    """새 프로필 생성
    - 프로필 이름은 고유해야 함

    Args:
        req (CreateProfileReq): 프로필 생성 요청 스키마

    Returns:
        meta (Dict[str, Any]): 생성된 프로필 메타데이터
    """
    meta = create_profile(req.name, req.description, req.head_info)
    return {"profile": meta.model_dump()}

@app.patch(
    "/profiles/{profile_id}",
    tags=["Profiles"],
    summary="프로필 메타 업데이트(부분 수정)",
    response_model=Dict[str, Any],
)
def api_update_profile(profile_id: str, req: UpdateProfileReq):
    try:
        meta = update_profile(
            profile_id,
            name=req.name,
            description=req.description,
            head_info=req.head_info,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="프로필을 찾을 수 없습니다.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if hasattr(app.state, "profile_agents"):
        app.state.profile_agents.pop(profile_id, None)
    return {"profile": meta.model_dump()}

@app.delete(
    "/profiles/{profile_id}",
    tags=["Profiles"],
    summary="프로필 삭제(예시/벡터스토어 포함)",
    response_model=Dict[str, Any],
)
def api_delete_profile(profile_id: str, drop_vectorstore: bool = True):
    ok = delete_profile(profile_id, drop_vectorstore=drop_vectorstore)
    if not ok:
        raise HTTPException(status_code=404, detail="프로필을 찾을 수 없습니다.")
    # 캐시 무효화
    if hasattr(app.state, "profile_agents"):
        app.state.profile_agents.pop(profile_id, None)
    return {"deleted": True}


@app.get(
    "/profiles",
    tags=["Profiles"],
    summary="모든 프로필 조회",
    description="저장된 모든 프로필의 메타데이터를 조회합니다.",
    response_model=Dict[str, Any],
    response_description="프로필 메타데이터 리스트",
)
def api_list_profiles():
    """모든 프로필 메타데이터 조회

    Returns:
        metas (Dict[str, Any]): 프로필 메타데이터 리스트
    """
    metas = [m.model_dump() for m in list_profiles()]
    return {"profiles": metas}


@app.get(
    "/profiles/{profile_id}",
    tags=["Profiles"],
    summary="특정 프로필 조회",
    description="**profile_id** 에 해당하는 프로필 메타데이터를 조회합니다.",
    response_model=Dict[str, Any],
    response_description="프로필 메타데이터",
)
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


ALLOWED_EXTS = {".txt", ".md"}
MAX_SIZE = 2 * 1024 * 1024


def _safe_read_text(file: UploadFile) -> Optional[str]:
    import os

    _, ext = os.path.splitext(file.filename or "")
    ext = ext.lower()
    if ext not in ALLOWED_EXTS:
        return None
    data = file.file.read(MAX_SIZE + 1)
    if len(data) > MAX_SIZE:
        return None
    return data.decode("utf-8", errors="ignore").strip() or None

@app.post(
    "/profiles/{profile_id}/examples",
    tags=["Profiles"],
    summary="프로필에 예시 텍스트 추가",
    description="지정한 **profile_id** 의 프로필에 예시 텍스트들을 추가합니다.",
    response_model=Dict[str, Any],
    response_description="업데이트된 프로필 메타데이터",
    operation_id="add_examples",
)
async def api_add_examples(
    profile_id: str,
    texts: Optional[List[str]] = Form(
        None, description="예시 텍스트(여러 개 가능)"
    ),
    files: Optional[List[UploadFile]] = File(None, description="예시 파일(한 개 이상 가능)")
):
    """프로필에 예시 텍스트 추가

    Args:
        profile_id (str): 프로필 ID

    Returns:
        meta (Dict[str, Any]): 업데이트된 프로필 메타데이터
    """
    try:
        collected: List[str] = []
        texts_list: List[str] = [t.strip() for t in texts if t and t.strip()]
        files_list: List[UploadFile] = files
        if texts_list:
            collected += [t.strip() for t in texts_list if t and t.strip()]
        for f in files_list:
            content = _safe_read_text(f)
            if content:
                collected.append(content)
        if not collected:
            raise HTTPException(
                status_code=400, detail="추가할 예시 텍스트가 없습니다."
            )
        meta = add_examples(profile_id, collected)

        if hasattr(app.state, "profile_agents"):
            app.state.profile_agents.pop(profile_id, None)

        return {"profile": meta.model_dump()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="프로필을 찾을 수 없습니다.")


@app.post(
    "/profiles/{profile_id}/train",
    tags=["Profiles"],
    summary="프로필 학습 수행",
    description="프로필 메타데이터와 예시를 바탕으로 학습을 수행합니다.",
    response_model=Dict[str, Any],
    response_description="업데이트된 프로필 메타데이터",
)
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
        raise HTTPException(status_code=404, detail="프로필을 찾을 수 없습니다.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"profile": meta.model_dump()}


@app.post(
    "/profiles/{profile_id}/complete",
    tags=["Profiles"],
    summary="프로필 스타일을 적용해 완성 글 생성",
    description=(
        "지정한 **profile_id** 의 스타일을 적용하여 초안을 보완/완성합니다.\n"
        "- `user_draft`: 사용자 초안 텍스트\n"
        "- `retriever_k`: 검색 문서 수"
    ),
    response_model=Dict[str, Any],
    response_description="완성 결과",
)
def api_complete_with_profile(profile_id: str, req: CompleteReq, request: Request) -> Dict[str, Any]:
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
            cache=getattr(request.app.state, "profile_chain_cache", None),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="프로필을 찾을 수 없습니다.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"output": out}
