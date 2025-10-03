from fastapi import APIRouter, HTTPException, File, Form, UploadFile, Request
from typing import Dict, Any, List, Optional

from note_agent.core.profiles import (
    create_profile,
    update_profile,
    delete_profile,
    list_profiles,
    add_examples,
    train_profile,
    load_profile,
    complete_with_profile,
)
from note_agent.model import (
    CompleteReq,
    CreateProfileReq,
    UpdateProfileReq,
    NoteAgentOutput,
)

router = APIRouter(tags=["Profiles"])

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


@router.post(
    "/profiles",
    summary="새 프로필 생성",
    response_model=Dict[str, Any],
    response_description="생성된 프로필 메타데이터를 반환합니다.",
)
async def api_create_profile(req: CreateProfileReq):
    meta = create_profile(req.name, req.description, req.head_info)
    return {"profile": meta.model_dump()}


@router.patch(
    "/profiles/{profile_id}",
    summary="프로필 메타 업데이트(부분 수정)",
    response_model=Dict[str, Any],
)
async def api_update_profile(profile_id: str, req: UpdateProfileReq):
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
    return {"profile": meta.model_dump()}


@router.delete(
    "/profiles/{profile_id}",
    summary="프로필 삭제(예시/벡터스토어 포함)",
    response_model=Dict[str, Any],
)
async def api_delete_profile(profile_id: str, drop_vectorstore: bool = True):
    ok = delete_profile(profile_id, drop_vectorstore=drop_vectorstore)
    if not ok:
        raise HTTPException(status_code=404, detail="프로필을 찾을 수 없습니다.")
    return {"deleted": True}


@router.get(
    "/profiles",
    summary="모든 프로필 조회",
    description="저장된 모든 프로필의 메타데이터를 조회합니다.",
    response_model=Dict[str, Any],
    response_description="프로필 메타데이터 리스트",
)
async def api_list_profiles():
    metas = [m.model_dump() for m in list_profiles()]
    return {"profiles": metas}


@router.get(
    "/profiles/{profile_id}",
    summary="특정 프로필 조회",
    description="**profile_id** 에 해당하는 프로필 메타데이터를 조회합니다.",
    response_model=Dict[str, Any],
    response_description="프로필 메타데이터",
)
async def api_get_profile(profile_id: str):
    try:
        meta = load_profile(profile_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"profile": meta.model_dump()}


@router.post(
    "/profiles/{profile_id}/examples",
    summary="프로필에 예시 텍스트 추가",
    description="지정한 **profile_id** 의 프로필에 예시 텍스트들을 추가합니다.",
    response_model=Dict[str, Any],
    response_description="업데이트된 프로필 메타데이터",
    operation_id="add_examples",
)
async def api_add_examples(
    profile_id: str,
    texts: Optional[List[str]] = Form(None, description="예시 텍스트(여러 개 가능)"),
    files: Optional[List[UploadFile]] = File(
        None, description="예시 파일(한 개 이상 가능)"
    ),
):
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
        return {"profile": meta.model_dump()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="프로필을 찾을 수 없습니다.")


@router.post(
    "/profiles/{profile_id}/train",
    summary="프로필 학습 수행",
    description="프로필 메타데이터와 예시를 바탕으로 학습을 수행합니다.",
    response_model=Dict[str, Any],
    response_description="업데이트된 프로필 메타데이터",
)
async def api_train_profile(profile_id: str):
    try:
        meta = train_profile(profile_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="프로필을 찾을 수 없습니다.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"profile": meta.model_dump()}


@router.post(
    "/profiles/{profile_id}/complete",
    summary="프로필 스타일을 적용해 완성 글 생성",
    description=(
        "지정한 **profile_id** 의 스타일을 적용하여 초안을 보완/완성합니다.\n- `user_draft`: 사용자 초안 텍스트\n- `retriever_k`: 검색 문서 수"
    ),
    response_model=Dict[str, Any],
    response_description="완성 결과",
)
async def api_complete_with_profile(
    profile_id: str, req: CompleteReq, request: Request
) -> Dict[str, Any]:
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
