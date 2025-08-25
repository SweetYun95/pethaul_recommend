
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

query="""
SELECT * FROM likes
"""

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


data = pd.read_sql(query,engine)
# print('=======data=======',data)

df=data[['userId','itemId']]
print('=======df=======',df)

user_en=LabelEncoder()
item_en=LabelEncoder()

df['user_id_enc']=user_en.fit_transform(df['userId'])
df["item_id_enc"]=item_en.fit_transform(df['itemId'])

matrix = csr_matrix(
    (np.ones(len(df)), (df["user_id_enc"], df["item_id_enc"]))
)


model = AlternatingLeastSquares(factors=10, iterations = 50)
model.fit(matrix)



@app.get("/recommend")
def recommend(user_id: int = Query(..., description="원본 user_id 입력"), top_n: int = 10):
    if user_id not in df['userId'].values:
        raise HTTPException(status_code=404, detail="해당 user_id는 데이터에 없습니다.")
    
    user_idx=user_en.transform([user_id])
    user_v = csr_matrix(matrix[user_idx])

    item_indices , scores = model.recommend(
        userid = user_idx[0],
        user_items=user_v,
        N=top_n,
    )
    print('=====  item_indices:',  item_indices)
    print('===== scores:', np.round((scores*100),3)) 

    item_de = item_en.inverse_transform(item_indices)
    print('===== result:', item_de)

    res = [{'item':int(item_id), 'score':float(score)}
           for item_id, score in zip (item_de, scores)]
    
    return res 

recommend(3)