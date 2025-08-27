# 추천 시스템 서비스 (Implicit ALS 기반)

이 문서는 `FastAPI` + `implicit` 라이브러리로 구현된 아이템 추천 API의 동작 원리와 실행 방법을 설명합니다. 본 코드는 **사용자의 ‘좋아요(likes)’ 기록**을 기반으로 한 **암묵적피드백(Implicit Feedback)** 협업필터링 모델(ALS)을 학습하여, 사용자가 선호할 확률이 높은 아이템을 추천합니다.

---

## 핵심 요약

* **입력 데이터**: MySQL `likes` 테이블의 `(userId, itemId)` 쌍
* **모델**: `implicit.als.AlternatingLeastSquares` (요인 수 `factors=10`, 반복 `iterations=50`)
* **행렬 구축**: `(사용자, 아이템)`의 희소 행렬 `CSR` (모든 관측의 가중치는 1)
* **엔드포인트**: `GET /recommend?user_id=<원본ID>&top_n=<개수>`
* **출력**: 추천 아이템 리스트 `[{ item: <itemId>, score: <float> }, ...]`

---

## 시스템 구성

### 1) 의존성

* Python 3.10+
* FastAPI, Uvicorn (ASGI 서버)
* pandas, numpy, scikit-learn (LabelEncoder)
* scipy (sparse matrix)
* implicit (ALS)
* SQLAlchemy + PyMySQL (MySQL 접속)
* python-dotenv (환경변수 로드)

### 2) 환경변수 (.env)

다음 키를 `.env`에 설정합니다.

```env
DB_USER=...
DB_PASSWORD=...
DB_HOST=...
DB_PORT=3306
DB_NAME=...
FRONTEND_APP_URL=http://localhost:5173
```

* `FRONTEND_APP_URL`은 CORS 허용 Origin으로 사용됩니다.

### 3) 데이터 스키마 (가정)

`likes` 테이블에 최소한 아래 컬럼이 존재해야 합니다.

```sql
SELECT userId, itemId FROM likes;
```

* `userId`: 정수형 사용자 식별자
* `itemId`: 정수형 아이템 식별자

---

## 동작 원리

1. **데이터 적재**: `SELECT * FROM likes`로 전체 좋아요 데이터를 읽어 `(userId, itemId)`를 추출합니다.
2. **ID 인코딩**: `LabelEncoder`로 `userId`, `itemId`를 0..N-1 정수 인덱스로 변환합니다.
3. **희소 행렬 생성**: `(row=user_idx, col=item_idx, data=1)`로 CSR 행렬을 만듭니다.
4. **모델 학습**: Implicit ALS를 사용해 사용자 잠재벡터와 아이템 잠재벡터를 학습합니다.
5. **추천 생성**: 주어진 사용자에 대해 `model.recommend()`로 상위 `N`개 아이템과 점수를 계산합니다.
6. **역인코딩**: 예측된 아이템 인덱스를 원본 `itemId`로 복원하여 응답합니다.

> **Implicit ALS란?**
> 명시적 평점(예: 별점)이 아닌, 클릭/구매/좋아요 같은 행동 기록을 **선호 신호**로 보고 행렬분해를 수행하는 기법입니다. 이 구현에서는 모든 관측을 동일 가중치(1)로 처리합니다.

---

## API

### `GET /recommend`

**Query Params**

* `user_id` *(int, required)*: 원본 사용자 ID
* `top_n` *(int, optional, default=10)*: 추천 개수

**응답 형식**

```json
[
  { "item": 123, "score": 0.8471 },
  { "item": 456, "score": 0.8123 }
]
```

**에러**

* `404 Not Found`: `user_id`가 학습 데이터에 존재하지 않는 경우

**예시 요청**

```bash
curl -G "http://localhost:8000/recommend" \
  --data-urlencode "user_id=3" \
  --data-urlencode "top_n=10"
```

---

## 실행 방법

1. **패키지 설치**

```bash
pip install fastapi uvicorn python-dotenv pandas numpy scikit-learn scipy implicit sqlalchemy pymysql
```

2. **환경변수 설정**: `.env` 파일을 프로젝트 루트에 배치
3. **서버 실행**

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

> 파일명이 `app.py`가 아닐 경우 `uvicorn <파일명_without_.py>:app` 형태로 실행하세요.

---

## 코드 흐름 (요약)

```python
# 1) .env 로드, DB 연결
engine = create_engine("mysql+pymysql://...")

# 2) 데이터 로드
data = pd.read_sql("SELECT * FROM likes", engine)

# 3) 인코딩 및 희소행렬
user_en = LabelEncoder(); item_en = LabelEncoder()
df["user_id_enc"] = user_en.fit_transform(df["userId"])
df["item_id_enc"] = item_en.fit_transform(df["itemId"])
matrix = csr_matrix((np.ones(len(df)), (df["user_id_enc"], df["item_id_enc"])) )

# 4) 모델 학습
model = AlternatingLeastSquares(factors=10, iterations=50)
model.fit(matrix)

# 5) 추천 엔드포인트
@app.get("/recommend")
def recommend(user_id: int, top_n: int = 10):
    if user_id not in df["userId"].values:
        raise HTTPException(status_code=404, detail="해당 user_id는 데이터에 없습니다.")
    user_idx = user_en.transform([user_id])[0]
    user_items = csr_matrix(matrix[user_idx])
    item_indices, scores = model.recommend(userid=user_idx, user_items=user_items, N=top_n)
    item_ids = item_en.inverse_transform(item_indices)
    return [{"item": int(i), "score": float(s)} for i, s in zip(item_ids, scores)]
```

---

## 모델/데이터 관리 팁

* **재학습(리트레이닝)**: 현재 코드는 서버 시작 시점에 한 번 학습합니다. 데이터가 자주 바뀐다면 주기적 재학습(예: 스케줄러) 또는 온디맨드 트리거를 고려하세요.
* **가중치/신뢰도**: 모든 관측을 동일 가중치(1)로 처리 중입니다. 클릭<장바구니<구매와 같이 행동에 따라 다른 가중치를 줄 수도 있습니다.
* **냉시작(Cold Start)**: 신규 사용자/아이템은 협업필터링만으로 추천이 어렵습니다. 인기 기반 추천(Top-N), 신상품 부스팅, 콘텐츠 기반 모델과의 혼합을 고려하세요.
* **하이퍼파라미터**: `factors`, `iterations`, `regularization`, `alpha`(신뢰도 스케일) 등을 교차검증으로 조정하면 품질이 개선됩니다.
* **성능**: 대용량에서는 `CSR` 유지, `implicit`의 GPU/병렬 옵션 검토, Annoy/FAISS 같은 근사 NN을 활용한 서빙도 고려하십시오.

---

## 보안 & 운영

* **CORS**: `.env`의 `FRONTEND_APP_URL`만 허용합니다. 운영환경에서는 Origin 화이트리스트를 엄격히 관리하세요.
* **DB 비밀정보**: `.env`에 보관하고 저장소에 커밋하지 않습니다.
* **예외 처리**: 존재하지 않는 `user_id` 요청 시 404를 반환합니다. 기타 DB 연결 실패, 데이터 스키마 불일치 등은 로깅 후 5xx로 처리하는 것이 바람직합니다.

---

## 한계와 향후 개선

* 단일 시그널(좋아요)만 사용 → **다중 피드백**(조회/장바구니/구매/리뷰점수) 통합 고려
* 사용자/아이템 메타데이터 미활용 → 콘텐츠 기반 특징 결합(하이브리드 추천)
* 배치 학습만 구현 → 온라인 학습/증분 업데이트 도입
* 평가 코드 미포함 → Hold-out/Time-split 기반의 MAP\@K, NDCG\@K 평가 파이프라인 추가

---
