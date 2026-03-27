import logging


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger customizado formatado para tracking limpo."""
    logger = logging.getLogger(name)

    # Previne handlers duplicados rodando scripts iterativamente
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()

        # Formato limpo e objetivo ISO8601 level msg
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.propagate = False  # isenção do root logger

    return logger
