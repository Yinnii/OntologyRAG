from pathlib import Path
from loguru import logger as _logger
from datetime import datetime

def get_ontorag_logger():
    logfile_level = "DEBUG"
    name: str = None
    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d")
    log_name = f"{name}_{formatted_date}" if name else formatted_date  # name a log with prefix name

    # _logger.remove()
    _logger.level("ONTORAG", color="<blue>", no=25)
    # _logger.add(sys.stderr, level=print_level)
    _logger.add(Path("logs") / f"{log_name}.txt", level=logfile_level)
    _logger.propagate = False
    return _logger

ontorag_logger = get_ontorag_logger()