from typing import Optional
from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session
from jose import jwt, JWTError

from app.database.session import get_db
from app.database.models import APIKey, User
from app.config import settings


def require_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> APIKey:
    key = (
        db.query(APIKey)
        .filter(APIKey.key == x_api_key, APIKey.is_active == True)
        .first()
    )
    if not key:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    return key


def optional_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> Optional[APIKey]:
    if not x_api_key:
        return None
    return (
        db.query(APIKey)
        .filter(APIKey.key == x_api_key, APIKey.is_active == True)
        .first()
    )


def get_current_user(
    authorization: str = Header(...),
    db: Session = Depends(get_db),
) -> User:
    credentials_error = HTTPException(status_code=401, detail="Invalid or expired token")
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            raise credentials_error
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        user_id: int = int(payload.get("sub"))
    except (JWTError, ValueError, AttributeError):
        raise credentials_error

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise credentials_error
    return user
