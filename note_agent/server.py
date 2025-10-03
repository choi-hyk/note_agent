from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from langserve import add_routes
from langchain_core.runnables import Runnable

from note_agent.model import NoteAgentInput, NoteAgentOutput
from note_agent.core.profiles import complete_with_profile, load_profile
from note_agent.api.profile_routers import router as api_profiles_router


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
            cache=cached,
        )

    async def _arun(user_draft: str, retriever_k: int | None):
        return _run(user_draft, retriever_k)

    app.state.profile_agents[profile_id] = (_run, _arun)
    return app.state.profile_agents[profile_id]


class NoteAgent(Runnable):
    """
    LangServe에 등록할 Runnable 구현.
    - 입력(JSON): {"profile_id": "...", "user_draft": "...", "user_input": "...", "head_info": "..." "retriever_k": 3}
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

        user_input = input.get("user_input") or ""
        if not user_input:
            raise ValueError("user_input은 필수입니다.")

        head_info = input.get("head_info", None)

        retriever_k = input.get("retriever_k", 3)

        _ensure_profile_exists(profile_id)

        run, _ = _ensure_profile_agent(profile_id)
        return run(
            user_draft=user_draft,
            user_input=user_input,
            head_info=head_info,
            retriever_k=retriever_k,
        )

    async def ainvoke(self, input: dict, config=None):
        if not isinstance(input, dict):
            raise ValueError("input은 객체(JSON)여야 합니다.")

        profile_id = input.get("profile_id")
        if not profile_id:
            raise ValueError("profile_id는 필수입니다.")

        user_draft = input.get("user_draft") or ""
        if not user_draft:
            raise ValueError("user_draft는 필수입니다.")

        user_input = input.get("user_input") or ""
        if not user_input:
            raise ValueError("user_input은 필수입니다.")

        head_info = input.get("head_info", None)

        retriever_k = input.get("retriever_k", 3)

        _ensure_profile_exists(profile_id)

        _, arun = _ensure_profile_agent(profile_id)
        return await arun(
            user_draft=user_draft,
            user_input=user_input,
            head_info=head_info,
            retriever_k=retriever_k,
        )

    def stream(self, input: dict, config=None):
        yield self.invoke(input, config)


note_agent = NoteAgent()
add_routes(
    app,
    note_agent,
    path="/note-agent",
    input_type=NoteAgentInput,
    output_type=NoteAgentOutput,
)

app.include_router(api_profiles_router)
