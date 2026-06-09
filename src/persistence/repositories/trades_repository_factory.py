"""
Фабрика для создания репозитория сделок.

Позволяет переключаться между SQLite и Supabase через конфигурацию.
"""

from __future__ import annotations

import logging

from src.persistence.repositories.base_trades_repository import BaseTradesRepository
from src.persistence.repositories.execution_trades_repository import (
    ExecutionTradesRepository,
)

logger = logging.getLogger(__name__)


def create_trades_repository(
    db_type: str = "sqlite",
    db_path: str | None = None,
    supabase_url: str | None = None,
    supabase_key: str | None = None,
) -> BaseTradesRepository:
    """
    Создает и возвращает репозиторий сделок на основе конфигурации.

    Args:
        db_type: Тип базы данных - "sqlite" или "supabase"
        db_path: Путь к SQLite файлу (для sqlite)
        supabase_url: URL Supabase проекта (для supabase)
        supabase_key: API ключ Supabase (для supabase)

    Returns:
        BaseTradesRepository: Реализация репозитория

    Raises:
        ValueError: Если неправильный db_type или отсутствуют необходимые параметры
        ImportError: Если supabase не установлен при db_type="supabase"
    """
    db_type = db_type.lower()
    logger.info(f"Создание репозитория типа: {db_type}")

    if db_type == "sqlite":
        if db_path is None:
            raise ValueError("db_path обязателен для типа 'sqlite'")
        logger.info(f"📁 SQLite репозиторий: {db_path}")
        return ExecutionTradesRepository(db_path)

    if db_type == "supabase":
        if supabase_url is None or supabase_key is None:
            raise ValueError("supabase_url и supabase_key обязательны для типа 'supabase'")
        from src.persistence.repositories.supabase_trades_repository import (
            SupabaseTradesRepository,
        )
        logger.info(f"☁️  Supabase репозиторий: {supabase_url}")
        return SupabaseTradesRepository(supabase_url, supabase_key)

    raise ValueError(f"Неизвестный тип базы данных: {db_type}. Используйте 'sqlite' или 'supabase'")


def create_trades_repository_from_config() -> BaseTradesRepository:
    """
    Создает репозиторий на основе настроек из config.py.

    Использует переменные окружения для чувствительных данных (Supabase ключ).
    """
    import os

    import config

    db_type = getattr(config, "EXECUTION_DB_TYPE", "sqlite")
    logger.info(f"EXECUTION_DB_TYPE = {db_type}")

    if db_type == "sqlite":
        db_path = getattr(config, "EXECUTION_DB_PATH", None) or config.DB_PATH
        return create_trades_repository(db_type="sqlite", db_path=db_path)

    if db_type == "supabase":
        supabase_url = getattr(config, "SUPABASE_URL", None)
        supabase_key = (
            os.getenv("SUPABASE_KEY")
            or os.getenv("SUPABASE_SERVICE_KEY")
            or getattr(config, "SUPABASE_KEY", None)
        )

        if supabase_url is None:
            raise ValueError(
                "SUPABASE_URL должен быть задан в config.py для использования Supabase"
            )
        if supabase_key is None:
            raise ValueError(
                "SUPABASE_KEY или SUPABASE_SERVICE_KEY должен быть задан "
                "в переменных окружения для использования Supabase"
            )

        return create_trades_repository(
            db_type="supabase",
            supabase_url=supabase_url,
            supabase_key=supabase_key,
        )

    raise ValueError(f"Неизвестный EXECUTION_DB_TYPE: {db_type}")
