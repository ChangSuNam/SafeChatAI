import coremltools as ct
import torch
import numpy as np
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
import shutil
import os

# Load and prepare the model
model_path = "fine_tuned_mobilebert"
tokenizer = MobileBertTokenizer.from_pretrained(model_path)
model = MobileBertForSequenceClassification.from_pretrained(model_path, num_labels=2)
model.eval()

class MobileBERTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Normalize logits, prevent overflow
        logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        return torch.softmax(logits, dim=-1)

wrapped_model = MobileBERTWrapper(model)
wrapped_model.eval()

# Trace the model
sample_text = "I like"
inputs = tokenizer(sample_text, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
input_ids = inputs["input_ids"].to(torch.int32)
attention_mask = inputs["attention_mask"].to(torch.int32)
traced_model = torch.jit.trace(wrapped_model, (input_ids, attention_mask))

# Converting to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, 64), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, 64), dtype=np.int32)
    ],
    outputs=[ct.TensorType(name="probabilities", dtype=np.float32)],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS15,
    compute_precision=ct.precision.FLOAT32,
    compute_units=ct.ComputeUnit.ALL
)


# Saving the model. Move the creataed file into the project package before running the app.

output_path = "FineTunedMobileBERT.mlpackage"
if os.path.exists(output_path):
    shutil.rmtree(output_path)
    
mlmodel.save(output_path)
