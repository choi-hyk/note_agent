import uuid
from typing import List, Dict, Any, Literal, Optional
from datetime import datetime
from pydantic import BaseModel

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
    length_info: ProfileLengthInfo = None
    head_info: Optional[List[ProfileHeadInfo]] = None
    persist_dir: Optional[str] = None
    examples_count: int = 0   