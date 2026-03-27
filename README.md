# Product Semantic Search API

Python(FastAPI) 기반 API 서버.  
**한 프로젝트** 안에 **패키지로 서비스를 구분**하며, **검색(agent)** 과 **동기화(manager)** 는 **각각 별도로 띄웁니다.**

| 서비스명 | 역할 | 망 | 포트 |
|----------|------|-----|------|
| **agent** | 검색 API (OpenSearch에서 상품 코드 조회) | public | **8090** |
| **manager** | 동기화 API (S3 → OpenSearch 등록) | private | **8091** |

## 프로젝트 구조 (정규화)

- **공통**: `app/core/` 에 설정·로깅·OpenSearch·프롬프트 로더 통합.
- **agent / manager**: 동일 계층 구조 — `main.py`, `routers/`, `services/` 로 일관.
- **프롬프트**: `app/prompts/<service>/<name>.txt`. 프로덕션에서 `PROMPTS_DIR` 로 외부 경로 지정 가능.

```
app/
  __init__.py
  core/                     # 공통 인프라
    __init__.py
    config.py                # 환경 설정 (get_settings)
    logging.py               # 로깅 (get_agent_logger, get_manager_logger, get_exception_location)
    opensearch.py            # get_client()
    prompt_loader.py         # load_prompt(service, name)
  models/                    # API·도메인 모델
    __init__.py
    search.py                # SearchQueryParams, SearchResponse
    sync.py                  # SyncStartResponse
    index_fields.py          # ProductIndexFields
  prompts/
    agent/
      query_classifier_system.txt
      query_reconstructor_system.txt
  agent/                     # 검색 서비스 (public, 8090)
    __init__.py
    main.py
    routers/
      __init__.py
      search.py              # GET /api/v1/search
    services/
      search_service.py
      text_normalizer.py
      morph_analyzer.py
      query_classifier.py
      query_reconstructor.py
      embedding_client.py
  manager/                   # 동기화 서비스 (private, 8091)
    __init__.py
    main.py
    routers/
      __init__.py
      sync.py                # POST /api/v1/sync/start
    services/
      s3_client.py
      embedding_client.py
      opensearch_indexer.py
      sync_service.py
scripts/
  run_agent.bat
  run_manager.bat
```

## 요구사항

- Python 3.10+
- OpenSearch (로컬 또는 AWS OpenSearch, k-NN 플러그인)
- AWS S3 (manager 동기화 사용 시)
- AWS Bedrock (Titan v2 임베딩, Nova Micro 질문 분류)
- 한글 띄어쓰기·정규화: `pykospacing`(띄어쓰기) + `kiwipiepy`(형태소 분석 후 재결합)

## 설치

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
copy .env.example .env
# .env 에 OpenSearch, S3 값 입력
```

**__pycache__ 생성 방지**  
`scripts/run_agent.bat`, `scripts/run_manager.bat` 에는 `PYTHONDONTWRITEBYTECODE=1` 이 설정되어 있어, 이 스크립트로 실행하면 `__pycache__` 가 생성되지 않습니다.  
직접 `uvicorn` 을 실행할 때도 비활성화하려면: `set PYTHONDONTWRITEBYTECODE=1` (Windows) 또는 `export PYTHONDONTWRITEBYTECODE=1` (Linux/macOS) 후 실행하세요.

## 서비스 실행 (따로 띄우기)

두 API는 **각각 다른 포트**로 실행합니다.

**1. agent (검색, public, 포트 8090)**

```bash
uvicorn app.agent.main:app --reload --host 0.0.0.0 --port 8090
# 또는 (Windows)
scripts\run_agent.bat
```

- API: http://localhost:8090  
- 문서: http://localhost:8090/docs  

**2. manager (동기화, private, 포트 8091)**

```bash
uvicorn app.manager.main:app --reload --host 0.0.0.0 --port 8091
# 또는 (Windows)
scripts\run_manager.bat
```

- API: http://localhost:8091  
- 문서: http://localhost:8091/docs  

---

## 1. agent — 검색 API (public, 8090)

사용자가 **검색어**를 입력하면 **5단계 프로세스**로 상품 검색이 이뤄집니다.

**상품 검색 5단계**  
1. **텍스트 정규화** — Unicode 정규화(`unicodedata` NFKC), 특수문자·이모지·중복 공백 제거  
2. **형태소 분석** — Kiwi/MeCab 명사 추출 (`.env` `MORPH_ENGINE=kiwi|mecab`)  
3. **질문 재구성** — AWS Bedrock LLM으로 검색에 적합한 질문 한 줄로 재구성 (수정 가능)  
4. **벡터 임베딩** — AWS Amazon Titan v2로 재구성된 질문 임베딩  
5. **OpenSearch k-NN 검색** — 임베딩 벡터로 k-NN 검색 후 상품 코드 반환  

응답의 `query`에는 재구성된 질문 텍스트가, `query_type`에는 `"knn"`이 담깁니다.

| 항목 | 내용 |
|------|------|
| 메서드/경로 | `GET /api/v1/search` |
| 쿼리 파라미터 | `q` (검색어, 필수), `size` (결과 수, 기본 50, 최대 200) |
| 응답 모델 | `SearchResponse`: `query`(재구성됨), `query_type`(knn), `product_codes`, `count` |
| 로그 | **콘솔** + **파일** `logs/agent.yyyy-mm-dd.log` |

**예시**

```bash
curl "http://localhost:8090/api/v1/search?q=노트북&size=10"
```

---

## 2. manager — 동기화(동작 시작) API (private, 8091)

**동작 시작** API를 호출하면 **S3 특정 경로**에서 상품 정보를 가져와 **AWS Titan v2**로 임베딩한 뒤 **OpenSearch**에 등록합니다.  
임베딩 대상 필드와 일반 텍스트 필드는 `ProductIndexFields` 모델로 관리합니다.

| 항목 | 내용 |
|------|------|
| 메서드/경로 | `POST /api/v1/sync/start` |
| 응답 모델 | `SyncStartResponse`: `status`, `indexed`, `total_fetched`, `errors` |
| 임베딩 | AWS Bedrock Amazon Titan Text Embeddings v2 (`embedding_fields`에 지정된 필드) |
| 로그 | 진행 상황은 **콘솔**과 **파일**(`logs/manager.yyyy-mm-dd.log`)에 동시 기록 |

**예시**

```bash
curl -X POST "http://localhost:8091/api/v1/sync/start"
```

S3 상품 파일 형식: JSON 배열 `[...]` 또는 JSONL (한 줄당 JSON). `product_code` 또는 `id` 필드 필요.

---

## API 모델 (필드 관리)

요청/응답 필드는 `app/models/` 에서 Pydantic 모델로 관리하며, `from app.models import ...` 로 사용합니다.

| 모델 | 서비스 | 용도 |
|------|--------|------|
| `SearchQueryParams` | agent | 검색 쿼리: `q`, `size` |
| `SearchResponse` | agent | 검색 응답: `query`, `query_type`, `product_codes`, `count` |
| `ProductIndexFields` | manager | 인덱스 필드: `embedding_fields`, `text_fields`, `embedding_vector_field` |
| `SyncStartResponse` | manager | 동기화 응답: `status`, `indexed`, `total_fetched`, `errors` |

---

## 환경변수 (.env)

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `OPENSEARCH_HOST` | OpenSearch URL | `http://localhost:9200` |
| `OPENSEARCH_INDEX` | 상품 인덱스 이름 | `products` |
| `OPENSEARCH_USER` | OpenSearch 사용자 (선택) | - |
| `OPENSEARCH_PASSWORD` | OpenSearch 비밀번호 (선택) | - |
| `S3_BUCKET` | 동기화 소스 S3 버킷 (manager용) | (필수) |
| `S3_PREFIX` | 상품 파일 경로 prefix | `products/` |
| `AWS_REGION` | AWS 리전 | `ap-northeast-2` |
| `BEDROCK_EMBED_MODEL_ID` | Bedrock 임베딩 모델 ID (manager) | `amazon.titan-embed-text-v2:0` |
| `EMBED_DIMENSIONS` | 임베딩 차원 (256/512/1024) | `1024` |
| `NOVA_RECONSTRUCT_MODEL_ID` | Nova Micro 모델 ID (agent 질문 재구성 LLM) | `amazon.nova-micro-v1:0` |
| `MORPH_ENGINE` | 형태소 분석 엔진 (kiwi \| mecab) | `kiwi` |
| `OPENSEARCH_EMBEDDING_FIELD` | k-NN 검색 벡터 필드명 | `embedding` |
| `PROMPTS_DIR` | 프롬프트 파일 루트 (미설정 시 앱 내장 `app/prompts` 사용) | (비어 있음) |
| `LOG_DIR` | 로그 디렉터리 | `logs` |
| `LOG_FILE_AGENT` | agent 로그 파일 접두사 → `agent.yyyy-mm-dd.log` | `agent` |
| `LOG_FILE_MANAGER` | manager 로그 파일 접두사 → `manager.yyyy-mm-dd.log` | `manager` |
| `LOG_LEVEL` | 로그 레벨 | `INFO` |

## 프로덕션 환경

- **프롬프트**: LLM용 시스템 프롬프트는 `app/prompts/<service>/<name>.txt` 에서 로드합니다. 배포 시 `PROMPTS_DIR` 을 설정하면 해당 경로를 우선 사용합니다. (예: `PROMPTS_DIR=/etc/product_semantic_search/prompts`)
- **로그**: `LOG_DIR`, `LOG_FILE_AGENT`, `LOG_FILE_MANAGER` 로 경로·파일명 제어.
- **설정**: OpenSearch, S3, Bedrock 등은 `.env` 또는 환경변수로 관리.
