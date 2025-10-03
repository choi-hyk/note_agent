import os
import uuid
import shutil
from typing import Any, Dict, List, Optional
from datetime import datetime

from sqlalchemy import (
    Tuple,
    create_engine,
    String,
    Integer,
    DateTime,
    Text,
    JSON,
    ForeignKey,
    select,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
    relationship,
)
from sqlalchemy.exc import IntegrityError

from note_agent.core.core import (
    build_or_load_vectorstore,
    summarize_style_rules,
    estimate_target_length,
    define_head_info,
    build_completion_chain,
    expand_to_min_length,
)
from note_agent.model import (
    NoteAgentOutput,
    ProfileMeta,
    ProfileLengthInfo,
    HeadInfo,
)
from note_agent.config import DB_URL, PERSIST_DIR, LLM_MODEL

# ============================================================
# SQLite 설정
# ============================================================
engine = create_engine(DB_URL, future=True, echo=False)
SessionLocal = sessionmaker(
    bind=engine, expire_on_commit=False, autoflush=False, future=True
)


def _persist_dir(profile_id: str) -> str:
    return os.path.join(PERSIST_DIR, profile_id)


# ============================================================
# ORM 모델
# ============================================================
class Base(DeclarativeBase):
    pass


class ProfileORM(Base):
    __tablename__ = "profiles"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    style_rules: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    length_info: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )  # ProfileLengthInfo.dict()
    head_info: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True
    )  # [ProfileHeadInfo.dict(), ...]

    persist_dir: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    examples_count: Mapped[int] = mapped_column(Integer, default=0)

    examples: Mapped[List["ProfileExampleORM"]] = relationship(
        back_populates="profile", cascade="all, delete-orphan"
    )


class ProfileExampleORM(Base):
    __tablename__ = "profile_examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"), index=True
    )
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    profile: Mapped[ProfileORM] = relationship(back_populates="examples")


Base.metadata.create_all(bind=engine)


# ============================================================
# 변환기 (ORM → Pydantic)
# ============================================================
def _orm_to_meta(p: ProfileORM) -> ProfileMeta:
    return ProfileMeta(
        profile_id=p.id,
        name=p.name,
        description=p.description or "",
        created_at=p.created_at.isoformat(),
        style_rules=p.style_rules,
        length_info=ProfileLengthInfo(**p.length_info) if p.length_info else None,
        head_info=(
            [HeadInfo(**h) for h in (p.head_info or [])] if p.head_info else None
        ),
        persist_dir=p.persist_dir,
        examples_count=p.examples_count,
    )


def create_profile(
    name: str, description: str, head_info: Optional[List[HeadInfo]]
) -> ProfileMeta:
    """프로필을 생성하는 함수

    Args:
        name (str): 프로필 이름
        description (str): 프로필 설명
        head_info (Optional[List[ProfileHeadInfo]]): 헤더 정보

    Returns:
        ProfileMeta: 생성된 프로필 메타데이터
    """
    profile_id = str(uuid.uuid4())
    persist_dir = _persist_dir(profile_id)

    head_info_json = (
        [h.model_dump() if hasattr(h, "model_dump") else dict(h) for h in head_info]
        if head_info
        else None
    )
    try:
        with SessionLocal.begin() as s:
            p = ProfileORM(
                id=profile_id,
                name=name,
                description=description,
                head_info=head_info_json,
                persist_dir=persist_dir,
            )
            s.add(p)
    except IntegrityError as e:
        raise ValueError(f"프로필 이름이 중복되었습니다: {name}") from e

    os.makedirs(persist_dir, exist_ok=True)

    with SessionLocal() as s:
        p = s.get(ProfileORM, profile_id)
        return _orm_to_meta(p)


def update_profile(
    profile_id: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    head_info: Optional[List[HeadInfo]] = None,
) -> ProfileMeta:
    """
    프로필 메타데이터 부분 수정(Partial Update)

    Args:
        profile_id: 수정할 프로필 ID
        name: (선택) 새 이름. 고유 제약(UNIQUE) 위반 시 ValueError 발생
        description: (선택) 새 설명
        head_info: (선택) 새 헤더 정보 리스트. None이면 변경 안 함

    Returns:
        ProfileMeta: 수정된 프로필 메타데이터

    Raises:
        FileNotFoundError: 프로필이 존재하지 않을 때
        ValueError: 이름 중복 등 제약 위반
    """
    with SessionLocal.begin() as s:
        p = s.get(ProfileORM, profile_id)
        if not p:
            raise FileNotFoundError("Profile not found")

        if name is not None and name != p.name:
            p.name = name
        if description is not None:
            p.description = description
        if head_info is not None:
            p.head_info = [
                (h.model_dump() if hasattr(h, "model_dump") else dict(h))
                for h in head_info
            ]

        try:
            s.add(p)
            s.flush()
        except IntegrityError as e:
            raise ValueError(f"프로필 이름이 중복되었습니다: {name}") from e

        return _orm_to_meta(p)


def delete_profile(
    profile_id: str,
    *,
    drop_vectorstore: bool = True,
) -> bool:
    """
    프로필 삭제(예시 포함 캐스케이드). 필요 시 벡터스토어 디렉토리도 삭제.

    Args:
        profile_id: 삭제할 프로필 ID
        drop_vectorstore: True면 persist_dir(Chroma 등)까지 제거

    Returns:
        bool: 실제로 삭제되면 True, 아니면 False

    Raises:
        None (존재하지 않으면 False 반환)
    """
    persist_dir: Optional[str] = None
    with SessionLocal() as s:
        p = s.get(ProfileORM, profile_id)
        if not p:
            return False
        persist_dir = p.persist_dir

    with SessionLocal.begin() as s:
        p = s.get(ProfileORM, profile_id)
        if not p:
            return False
        s.delete(p)

    if drop_vectorstore and persist_dir:
        try:
            shutil.rmtree(persist_dir, ignore_errors=True)
        except Exception:
            pass

    return True


def load_profile(profile_id: str) -> ProfileMeta:
    """
    프로필 id를 통해 메타데이터를 로드하는 함수

    Args:
        profile_id: 프로필 ID

    Returns:
        ProfileMeta: 프로필 메타데이터
    """
    with SessionLocal() as s:
        p = s.get(ProfileORM, profile_id)
        if not p:
            raise FileNotFoundError("프로필을 찾지 못했습니다.")
        return _orm_to_meta(p)


def list_profiles() -> List[ProfileMeta]:
    """모든 프로필 메타데이터 조회

    Returns:
        metas (List[ProfileMeta]): 프로필 메타데이터 리스트
    """
    with SessionLocal() as s:
        rows = s.scalars(
            select(ProfileORM).order_by(ProfileORM.created_at.desc())
        ).all()
        return [_orm_to_meta(p) for p in rows]


def add_examples(profile_id: str, texts: List[str]) -> ProfileMeta:
    """사용자 예시글(plain text) 저장

    Args:
        profile_id: 프로필 ID
        texts: 예시 텍스트 리스트

    Returns:
        metas (ProfileMeta): 업데이트된 프로필 메타데이터
    """
    clean_texts = [t.strip() for t in texts if t and t.strip()]
    if not clean_texts:
        return load_profile(profile_id)

    with SessionLocal.begin() as s:
        p = s.get(ProfileORM, profile_id)
        if not p:
            raise FileNotFoundError("Profile not found")

        for t in clean_texts:
            s.add(ProfileExampleORM(profile_id=profile_id, content=t))

        cnt = s.scalar(
            select(func.count())
            .select_from(ProfileExampleORM)
            .where(ProfileExampleORM.profile_id == profile_id)
        )
        p.examples_count = int(cnt or 0)
        s.add(p)
        s.flush()

        return _orm_to_meta(p)


def _load_examples_from_db(profile_id: str) -> List[str]:
    """디스크에서 예시글을 로드하는 함수

    Args:
        profile_id: 프로필 ID

    Returns:
        texts (List[str]): 예시 텍스트 리스트
    """
    with SessionLocal() as s:
        rows = s.scalars(
            select(ProfileExampleORM)
            .where(ProfileExampleORM.profile_id == profile_id)
            .order_by(ProfileExampleORM.created_at.asc(), ProfileExampleORM.id.asc())
        ).all()
        return [r.content for r in rows]


def train_profile(profile_id: str) -> ProfileMeta:
    """예시글 기반으로 벡터스토어/스타일 규칙/길이 정보 생성

    Args:
        profile_id: 프로필 ID

    Returns:
        ProfileMeta: 업데이트된 프로필 메타데이터
    """
    with SessionLocal.begin() as s:
        p = s.get(ProfileORM, profile_id)
        if not p:
            raise FileNotFoundError("Profile not found")

        examples = _load_examples_from_db(profile_id)
        if len(examples) < 1:
            raise ValueError("예시글이 최소 1개 이상 필요합니다. 먼저 업로드하세요.")
        os.makedirs(p.persist_dir or _persist_dir(profile_id), exist_ok=True)
        build_or_load_vectorstore(example_texts=examples, persist_dir=p.persist_dir)

        style_rules = summarize_style_rules(examples)
        length_info = estimate_target_length(examples)

        if not p.head_info:
            computed = define_head_info(examples)
            p.head_info = [h.model_dump() for h in computed]

        p.style_rules = style_rules
        p.length_info = length_info.model_dump()

        s.add(p)
        s.flush()

        return _orm_to_meta(p)


def complete_with_profile(
    profile_id: str,
    user_draft: str,
    *,
    retriever_k: int = 3,
    cache: Optional[Dict[str, Any]] = None,
) -> NoteAgentOutput:
    """프로필 스타일로 글 완성(TOC/본문/변경로그 반환)

    Args:
        profile_id: 프로필 ID
        user_draft: 사용자가 작성한 초안
        retriever_k: RAG 검색 시 상위 k개 문서 활용(기본 3)
        cache: 프로필 에이전트 캐시

    Returns:
        out (Dict[str, Any]): 완성 결과
    """
    chain = None
    meta = None
    if cache is not None:
        packed = cache.get(profile_id)
        if packed:
            chain, meta = packed

    if meta is None:
        meta = load_profile(profile_id)
        if not (meta.style_rules and meta.length_info and meta.persist_dir):
            train_profile(profile_id)
            meta = load_profile(profile_id)

    if chain is None:
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma

        embeddings = OpenAIEmbeddings()
        vs = Chroma(embedding_function=embeddings, persist_directory=meta.persist_dir)
        chain = build_completion_chain(
            style_rules=meta.style_rules,
            vs=vs,
            length_info=meta.length_info,
            retriever_k=retriever_k,
        )
        if cache is not None:
            cache[profile_id] = (chain, meta)

    result = chain.invoke(user_draft)

    if len(result.completed_text) < meta.length_info.min_chars:
        expanded = expand_to_min_length(
            text=result.completed_text,
            target_min=meta.length_info.min_chars,
            target_max=meta.length_info.max_chars,
            model=LLM_MODEL,
        )
        result.completed_text = expanded
        result.change_log.additions.append(
            f"최소 분량 미달로 사후 확장 수행(→ ≥{meta.length_info.min_chars}자)"
        )

    return NoteAgentOutput(
        toc=result.toc,
        completed_text=result.completed_text,
        change_log=result.change_log,
    )
