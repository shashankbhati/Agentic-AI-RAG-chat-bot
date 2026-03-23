import logging
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt

from app.database.session import get_db
from app.database.models import User, APIKey
from app.config import settings
from app.api.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Schemas ───────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    email: str
    api_key: str
    collection_name: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _hash_password(password: str) -> str:
    return pwd_context.hash(password)


def _verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def _create_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=7)
    return jwt.encode(
        {"sub": str(user_id), "exp": expire},
        settings.JWT_SECRET,
        algorithm="HS256",
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post("/auth/register", response_model=AuthResponse, summary="Register a new account")
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(email=body.email, password_hash=_hash_password(body.password))
    db.add(user)
    db.flush()  # get user.id before committing

    api_key = APIKey(user_id=user.id, name="Default")
    db.add(api_key)
    db.commit()
    db.refresh(user)
    db.refresh(api_key)

    logger.info(f"New user registered: {user.email}")
    return AuthResponse(
        access_token=_create_token(user.id),
        email=user.email,
        api_key=api_key.key,
        collection_name=api_key.collection_name,
    )


@router.post("/auth/login", response_model=AuthResponse, summary="Login and get access token")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not _verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    api_key = (
        db.query(APIKey)
        .filter(APIKey.user_id == user.id, APIKey.is_active == True)
        .order_by(APIKey.created_at)
        .first()
    )
    if not api_key:
        raise HTTPException(status_code=500, detail="No active API key found for this account")

    return AuthResponse(
        access_token=_create_token(user.id),
        email=user.email,
        api_key=api_key.key,
        collection_name=api_key.collection_name,
    )


@router.get("/auth/me", summary="Get current user info")
def me(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    keys = db.query(APIKey).filter(APIKey.user_id == current_user.id, APIKey.is_active == True).all()
    return {
        "email": current_user.email,
        "api_keys": [
            {"id": k.id, "name": k.name, "key": k.key, "collection_name": k.collection_name}
            for k in keys
        ],
    }
