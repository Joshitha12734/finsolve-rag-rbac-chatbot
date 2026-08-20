"""
Authentication and role assignment.

Uses HTTP Basic Auth (simple, stateless, well supported by both FastAPI
and Streamlit's `requests` calls). On success returns the user's record
(including their role) so the caller can apply RBAC filtering.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .config import USERS, hash_password

security = HTTPBasic()


def get_current_user(credentials: HTTPBasicCredentials = Depends(security)) -> dict:
    user = USERS.get(credentials.username)
    if user is None or user["password_hash"] != hash_password(credentials.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return {
        "username": credentials.username,
        "full_name": user["full_name"],
        "role": user["role"],
    }
