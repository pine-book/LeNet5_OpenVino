import numpy as np
import sys
from openvino.inference_engine import IENetwork, IECore

ie = IECore()
model = ie.read_model(ir_path)
compiled_model = ie.compile_model(model=model, device_name="CPU")