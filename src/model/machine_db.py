from sqlalchemy import create_engine, Column, String, DateTime, Integer, ForeignKey, JSON
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base  # 모듈 변경
from datetime import datetime  # datetime 모듈 임포트

Base = declarative_base()
class Machine_db(Base):
    __tablename__ = 'Machine'

    dataSetId = Column(String(50), primary_key=True)
    machineId = Column(String(50), primary_key=True)
    machineType = Column(String(50))
    machineDesc = Column(String(255))
    factoryId = Column(String(50))
    isUpdated = Column(DateTime, default=datetime.now)  # 기본값 설정
    isCreated = Column(DateTime, server_default='CURRENT_TIMESTAMP')
    deterioration_rate = Column(JSON, nullable=True)