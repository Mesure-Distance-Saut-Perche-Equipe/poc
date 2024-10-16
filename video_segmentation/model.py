from segment_anything import sam_model_registry, SamPredictor


class SegmentModel:
    def __init__(
        self,
        model_type="vit_h",
        model_path="",
        checkpoint="sam_vit_h_4b8939.pth",
        device="cpu",
    ):
        self.model_type = model_type
        self.model_path = model_path
        self.checkpoint = checkpoint
        self.device = device
        self.model = None
        self.predictor = None

    def load_model(self):
        """Loads the SAM model from the registry and sets it to the specified device."""
        self.model = sam_model_registry[self.model_type](
            checkpoint=self.model_path + self.checkpoint
        )
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)

    def get_predictor(self):
        """Returns the initialized predictor."""
        return self.predictor
