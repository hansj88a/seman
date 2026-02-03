# 프롬프트 파일 (서비스별 분리)

- **agent**: 검색 서비스용 LLM 프롬프트
  - `query_classifier_system.txt`: 질문 유형 분류 (NOUN / NATURAL_LANGUAGE)
  - `query_reconstructor_system.txt`: 질문 재구성 (검색용 한 줄)
- **manager**: (추가 시 동일 규칙)

프로덕션에서 수정 시: `.env`에 `PROMPTS_DIR=/path/to/prompts` 설정 후 동일 디렉터리 구조로 파일 배치.
