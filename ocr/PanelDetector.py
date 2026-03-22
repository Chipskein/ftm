
import logging
import os
import warnings
import torch
from transformers import AutoModel

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

def load_magi() -> AutoModel:
    logger.info("loading Magi v2 model...")
    model = AutoModel.from_pretrained(
        "ragavsachdeva/magiv2", trust_remote_code=True
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Magi loaded on GPU")
    else:
        logger.info("Magi loaded on CPU")
    return model