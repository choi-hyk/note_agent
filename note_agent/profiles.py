import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from sqlalchemy import (
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

from .core import (
    build_or_load_vectorstore,
    summarize_style_rules,
    estimate_target_length,
    define_head_info,
    build_completion_chain,
    expand_to_min_length,
    LLM_MODEL,
)

from .model import ProfileMeta, ProfileLengthInfo, ProfileHeadInfo

# ============================================================
# SQLite 설정
# ============================================================
DB_URL = os.getenv("DB_URL", "sqlite:///./note_agent.db")
engine = create_engine(DB_URL, future=True, echo=False)
SessionLocal = sessionmaker(
    bind=engine, expire_on_commit=False, autoflush=False, future=True
)


def _persist_dir(profile_id: str) -> str:
    base = os.getenv("RAG_BASE_DIR", "./rag_store")
    return os.path.join(base, profile_id)


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
            [ProfileHeadInfo(**h) for h in (p.head_info or [])] if p.head_info else None
        ),
        persist_dir=p.persist_dir,
        examples_count=p.examples_count,
    )


def create_profile(
    name: str, description: str, head_info: Optional[List[ProfileHeadInfo]]
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
