
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares
from sqlalchemy import create_engine

# .env 파일 로드
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# SQLAlchemy로 MySQL 연결
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

query = """ 
SELECT 
    o.userId AS user_id,
    oi.itemId AS item_id,
    CAST(SUM(oi.count) AS UNSIGNED) AS purchase_count
FROM orders o, orderitems oi
where o.id = oi.orderId
group by oi.itemId, o.userId
order by o.userId, oi.itemId;
"""

# data = pd.read_sql(query, engine)
# print(data)

app = FastAPI()

# 허용할 origin 설정
origins = [
    os.getenv("FRONTEND_APP_URL"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # 허용할 origin
    allow_credentials=True,         # 쿠키 인증 허용 여부
    allow_methods=["*"],            # 허용할 HTTP 메서드 (GET, POST 등)
    allow_headers=["*"],            # 허용할 HTTP 헤더
)


    # 구매기반 가상 데이터
data = pd.DataFrame({
    'user_id': [0, 1, 1, 2, 3, 3],
    'item_id': [101, 101, 102, 103, 102, 104],
    'purchase_count': [1, 1, 2, 1, 1, 1]
})



# 1. 정수형 인코딩
user_enc = LabelEncoder()
item_enc = LabelEncoder()
data['user_idx'] = user_enc.fit_transform(data['user_id'])
data['item_idx'] = item_enc.fit_transform(data['item_id'])

# 2. 행렬 데이터로 변환(implicit가 행렬 데이터로 학습하므로)
matrix = coo_matrix((data['purchase_count'], (data['user_idx'], data['item_idx'])))

user_item_matrix=matrix.tocsr() # tocsr 적용시 0이 아닌 값만 저장 (0값 포함시 메모리 낭비)
                                # implicit는 csr 적용된 데이터로만 학습이 가능

print('✨matrix:',matrix.toarray())
print('✨user_item_matrix:',user_item_matrix.toarray())

# 3. 데이터 학습
# factors = 10 : user, item을 10차원 벡터로 표현
# iterations = 15 : 15회 반복 학습 (과다 학습시 과적합, 최적해 도출 필수)
model = AlternatingLeastSquares(factors=10, iterations = 15)
model.fit(user_item_matrix)


# http://localhost:8000/recommend?user_id=1&top_n=3
@app.get("/recommend")
 # 파라미터 [user_id: 유저 id], [top_n: 상위 n개 제한(limit)]
    # user id의 query결과를 학습하고 추천 상품을 top_n개만큼 출력하는 def함수 
def recommend(user_id: int = Query(..., description="원본 user_id 입력 (예: 3)"), top_n: int = 3):
     # user_id Query(..., _): 필수입력값을 의미하며, 디폴트 값을 가져올 수 있음.
        # description: 해당 인자에 대한 설명
        # top_n값은 필수값이 아니며, 별도로 지정하지 않는 경우 초기값 3


    # 유저가 존재하지 않는 경우 예외처리
    if user_id not in data['user_id'].values:
        raise HTTPException(status_code=404, detail="해당 user_id는 데이터에 없습니다.")

    # user_id를 인코딩한 user_idx값 구하기
    user_idx = user_enc.transform([user_id]) 
    print('===== user_idx:', user_idx)

    # user 벡터값 구하기 (매트릭스)
    user_v = csr_matrix(user_item_matrix[user_idx]) # 특정 유저의 벡터값 csr형태로 변환
    print('===== user_v:', user_v)

    # 추천 결과 추출
    item_indices, scores = model.recommend(
        userid=user_idx[0],
        user_items = user_v,
        N=top_n
    )
    print('=====  item_indices:',  item_indices)
    print('===== scores:', np.round(scores,2))

    # 결과 디코딩
    item_d = item_enc.inverse_transform(item_indices)
    print ('디코딩 결과:', item_d)

    

    # 4. 특정 유저에게 추천하는 item id를 리턴
    res = [
        {'id': int(item_id), 'score':float(score)}
        for item_id, score in zip(item_d, scores)
    ]
    
    return res
