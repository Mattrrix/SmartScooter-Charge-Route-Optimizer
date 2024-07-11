from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Location(Base):
    __tablename__ = 'locations'
    id = Column(String, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    charge = Column(Float, default=100)
    type = Column(String)
    battery_count = Column(Integer, default=0)

class Edge(Base):
    __tablename__ = 'edges'
    id = Column(Integer, primary_key=True, autoincrement=True)
    from_id = Column(String, ForeignKey('locations.id'))
    to_id = Column(String, ForeignKey('locations.id'))
    weight = Column(Float)

engine = create_engine('sqlite:///scooters.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
