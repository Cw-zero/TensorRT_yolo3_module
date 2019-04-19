from abc import ABC, abstractmethod

class BaseModule(ABC):

	@abstractmethod
	def process_frame(self):
		pass

	@abstractmethod
	def process_frame_batch(self):
		pass