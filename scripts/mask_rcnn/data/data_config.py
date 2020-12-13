from mrcnn.config import Config


class ISRLHumanConfig(Config):
	NAME = "human_cfg"
	NUM_CLASSES = 1 + 1
	STEPS_PER_EPOCH = 180