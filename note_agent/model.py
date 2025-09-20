from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ------------------------------
# Profiles Models
# ------------------------------
class ProfileLengthInfo(BaseModel):
    """프로필 예시글 길이 정보

    Attributes:
        avg_chars (int): 평균 길이
        min_chars (int): 최소 길이
        max_chars (int): 최대 길이

    """

    avg_chars: int
    min_chars: int
    max_chars: int


class ProfileHeadInfo(BaseModel):
    """프로필 예시글 헤더 정보

    Attributes:
    level (Literal): 헤더 레벨
    title (str): 헤더 제목
    """

    level: Literal["H1", "H2", "H3", "H4"]
    title: str


class ProfileMeta(BaseModel):
    """
    프로필 메타데이터

    Attributes:
        profile_id (str): 프로필 ID
        name (str): 프로필 이름
        description (str): 프로필 설명
        created_at (str): 생성 일시(ISO 포맷)
        style_rules (Optional[str]): 요약된 스타일 규칙
        length_info (Optional[ProfileLengthInfo]): 길이 정보(최소/최대 문자 수)
        head_info (Optional[List[ProfileHeadInfo]]): 헤드 정보 (예시글 헤더 정보)
        persist_dir (Optional[str]): 벡터스토어 영속화 디렉토리
        examples_count (int): 저장된 예시글 수
    """

    profile_id: str
    name: str
    description: str
    created_at: str
    style_rules: Optional[str] = None
    length_info: Optional[ProfileLengthInfo] = None
    head_info: Optional[List[ProfileHeadInfo]] = None
    persist_dir: Optional[str] = None
    examples_count: int = 0


# ------------------------------
# Request / Response Models
# ------------------------------
class CreateProfileReq(BaseModel):
    """프로필 생성 요청

    Attributes:
        name (str): 프로필 이름
    """

    name: str
    description: str
    head_info: Optional[List[ProfileHeadInfo]] = None


class CompleteReq(BaseModel):
    """프로필을 적용하여 완성글 요청

    Attributes:
        user_draft (str): 사용자 초안
        retriever_k (int): 검색할 문서 수
    """

    user_draft: str
    retriever_k: int = 3


# =========================================
# Note Agent Models
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
