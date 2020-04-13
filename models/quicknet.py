from base.base_model import BaseModel
from larq_zoo.sota import QuickNet as LarqQuickNet


class QuickNet(BaseModel):
    def __init__(self, config):
        super(QuickNet, self).__init__(config)
        self.build_model()

    def build_model(self) -> None:
        """

        """
        self.model = LarqQuickNet(
            input_shape=(160 * 160),
        )
