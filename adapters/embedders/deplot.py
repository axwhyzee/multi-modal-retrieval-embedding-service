import logging
from io import BytesIO
from typing import List

from PIL import Image
from transformers import (  # type: ignore
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor,
)

from adapters.embedders.base import PlotModel, TextModel

logger = logging.getLogger(__name__)


class DePlotModel(PlotModel):

    def __init__(self, text_model: TextModel):
        logger.info("Initializing google/deplot")
        self._processor = Pix2StructProcessor.from_pretrained("google/deplot")
        self._deplot_model = (
            Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
        )
        self._text_model = text_model

    def embed(self, data: bytes) -> List[float]:
        image = Image.open(BytesIO(data))
        inputs = self._processor(
            images=image,
            text="Generate underlying data table of the figure below:",
            return_tensors="pt",
        )
        predictions = self._deplot_model.generate(**inputs, max_new_tokens=512)
        table_str = self._processor.decode(
            predictions[0], skip_special_tokens=True
        )
        return self._text_model.embed(table_str.encode("utf-8"))
