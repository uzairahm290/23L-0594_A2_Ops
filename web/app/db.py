"""Task 5 - every prediction leaves a permanent row: what was decided, when, by which
model, and how confidently. The image itself is never stored, only its SHA-256."""
import time
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, Integer, String, create_engine, func,
                        select, text)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_log"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=lambda: datetime.now(timezone.utc), index=True)
    filename = Column(String(255), nullable=False)          # sanitised name only
    image_sha256 = Column(String(64), nullable=False, index=True)
    width = Column(Integer)
    height = Column(Integer)
    label = Column(String(20), nullable=False)
    p_ai = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    model_name = Column(String(120), nullable=False)
    model_version = Column(String(40), nullable=False)
    backend = Column(String(30), nullable=False)
    latency_ms = Column(Integer)


class Store:
    def __init__(self, url: str, retries: int = 10):
        self.engine = create_engine(url, pool_pre_ping=True, future=True)
        # Compose already waits for the DB health check; the retry is a second safety net
        for attempt in range(retries):
            try:
                Base.metadata.create_all(self.engine)
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def add(self, **row) -> PredictionLog:
        with self.Session.begin() as s:
            rec = PredictionLog(**row)
            s.add(rec)
        return rec

    def page(self, page: int, per_page: int = 20):
        with self.Session() as s:
            total = s.scalar(select(func.count(PredictionLog.id)))
            rows = s.scalars(select(PredictionLog).order_by(PredictionLog.id.desc())
                             .offset((page - 1) * per_page).limit(per_page)).all()
            return rows, total

    def stats(self):
        with self.Session() as s:
            q = select(PredictionLog.label, func.count(), func.avg(PredictionLog.confidence)).group_by(PredictionLog.label)
            return {lab: (n, avg) for lab, n, avg in s.execute(q)}

    def ping(self) -> bool:
        try:
            with self.engine.connect() as c:
                c.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
