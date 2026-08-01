import logging
import sys


SUMMARY_LOGGER_NAME = "awesome_cil.summary"
DETAIL_LOGGER_NAME = "awesome_cil.detail"


# Configuration values are implementation identifiers.  Keep them stable for
# model dispatch, but write experiment logs under the method names used by the
# original papers and the baseline workbook.
METHOD_LOG_DIRECTORIES = {
    "adam_adapter": "APER",
    "bilora": "BiLoRA",
    "cllora": "CL-LoRA",
    "coda_prompt": "CODA-Prompt",
    "coil": "COIL",
    "der": "DER",
    "dualprompt": "DualPrompt",
    "ease": "EASE",
    "finetune": "Full Fine-Tuning",
    "foster": "FOSTER",
    "hidep": "HiDeP",
    "icarl": "iCaRL",
    "inflora": "InfLoRA",
    "l2p": "L2P",
    "lae": "LAE",
    "mos": "MOS",
    "sdlora": "SD-LoRA",
    "simplecil": "SimpleCIL",
}


def canonical_log_dir(model_name):
    """Return the paper-facing directory name for a model identifier."""
    return METHOD_LOG_DIRECTORIES.get(str(model_name).lower(), str(model_name))


def configure_logging(log_file):
    formatter = logging.Formatter("%(asctime)s [%(filename)s] => %(message)s")

    summary_logger = logging.getLogger(SUMMARY_LOGGER_NAME)
    detail_logger = logging.getLogger(DETAIL_LOGGER_NAME)

    for logger in (summary_logger, detail_logger):
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False

    summary_handler = logging.FileHandler(filename=log_file, mode="w")
    summary_handler.setFormatter(formatter)
    summary_logger.addHandler(summary_handler)

    detail_handler = logging.StreamHandler(sys.stdout)
    detail_handler.setFormatter(formatter)
    detail_logger.addHandler(detail_handler)


def _format_message(message, *args):
    if args:
        return message % args

    return str(message)


def log_summary(message, *args):
    logging.getLogger(SUMMARY_LOGGER_NAME).info(
        _format_message(message, *args), stacklevel=2
    )


def log_detail(message, *args):
    logging.getLogger(DETAIL_LOGGER_NAME).info(
        _format_message(message, *args), stacklevel=2
    )


def log_both(message, *args):
    formatted = _format_message(message, *args)
    logging.getLogger(SUMMARY_LOGGER_NAME).info(formatted, stacklevel=2)
    logging.getLogger(DETAIL_LOGGER_NAME).info(formatted, stacklevel=2)


def log_both_multiline(message):
    formatted = str(message)
    print(formatted)
    for line in formatted.splitlines():
        logging.getLogger(SUMMARY_LOGGER_NAME).info(line, stacklevel=2)


def print_and_log_summary(message):
    formatted = _format_message(message)
    print(formatted)
    logging.getLogger(SUMMARY_LOGGER_NAME).info(formatted, stacklevel=2)


def print_and_log_summary_multiline(message):
    formatted = str(message)
    print(formatted)
    for line in formatted.splitlines():
        logging.getLogger(SUMMARY_LOGGER_NAME).info(line, stacklevel=2)
