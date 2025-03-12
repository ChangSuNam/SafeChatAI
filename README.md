# SafeChatAI: Real Time AI Safety Classification on iOS

## Project Overview
SafeChatAI is a proof-of-concept iOS application demonstrating real-time text safety classification using a fine-tuned MobileBERT model, integrated via CoreML. This project showcases:
- **AI Safety**: Detects "safe" or "unsafe" text inputs, with a pre-inference regex-based guardrail to filter profanity, aligning with responsible AI deployment.
- **On-Device Efficiency**: Fine-tunes MobileBERT with 15 epochs on a custom dataset, converts it to CoreML with FLOAT32 precision, and runs inference on iOS with CPU-only configuration and Swift concurrency.
- **Full-Stack Integration**: Combines Python-based training and model conversion with a SwiftUI interface, demonstrating end-to-end ML deployment.

### Features
- Fine-tunes MobileBERT on a custom safety dataset for binary classification.
- Converts the model to CoreML's `.mlpackage` format with traced, normalized outputs.
- Offers a SwiftUI app with real-time classification, profanity filtering, and asynchronous processing.

##Project Components

### `fine_tune_mobile_bert.py`
- **Purpose**: Fine-tunes a pre-trained MobileBERT model for binary safety classification ("safe" vs. "unsafe").
- **Details**: Loads `google/mobilebert-uncased` and fine-tunes it on `safety_dataset.csv` using the Hugging Face `Trainer` API. Key settings are:
  - 15 epochs, 8-sample batch size, 5e-5 learning rate, and 500 warmup steps.
  - MPS acceleration (if available) or CPU fallback.
  - Saves the best model (lowest eval loss) to `"fine_tuned_mobilebert"`, with tokenizer.

### `convert_mobilebert.py`
- **Purpose**: Converts the fine-tuned MobileBERT model to CoreML for iOS deployment.
- **Details**: Wraps the model in `MobileBERTWrapper` to normalize logits and apply softmax, traces it with `torch.jit.trace` using "I like" (1x64 input), and exports to `FineTunedMobileBERT_New.mlpackage` via `coremltools`. Configures:
  - Inputs: `input_ids` and `attention_mask` (1x64, int32).
  - Output: `probabilities` (float32).
  - FLOAT32 precision, all compute units.

### `safety_dataset.csv`
- **Purpose**: Training data for fine-tuning.
- **Details**: A dataset with text samples labeled "safe" or "unsafe" (e.g., "I like" → safe, "I hate" → unsafe), used by `fine_tune_mobile_bert.py`.

### `ContentView.swift`
- **Purpose**: Implements the iOS app’s UI and inference logic.
- **Details**:
  - Uses `VocabularyManager` to load `vocab.txt` and tokenize inputs, matching MobileBERT’s `[CLS]`, `[SEP]`, `[UNK]` scheme.
  - Applies a regex guardrail to block profanity (e.g., "fuck", "shit", "damn").
  - Runs inference asynchronously with `async/await` and `DispatchQueue.global`, classifying text based on `probabilities` output.
  - Displays results (e.g., "This is a safe reply!" or "Input blocked...").

### `SafeChatAIApp.swift`
- **Purpose**: Serves as the app’s entry point.

### `vocab.txt`
- **Purpose**: Vocabulary file for tokenization. Generated by fine_tune_mobile_bert. Bundled in SafeChatAIApp. Included in the commits for simplicity.

---

## Development Environment Setup

### Prerequisites
- **macOS**: 14.0+ (tested on Sequoia)
- **Xcode**: 16.0+ (used iOS 18.2 simulator)
- **Python**: 3.10+


### Python Environment
1. Clone the repo.

2. Create and activate a virtual environment:
    **"python3 -m venv venv**
    **source venv/bin/activate**

3. Install dependencies:
    **pip install -r requirements.txt**
    
    (torch==2.1.0: PyTorch for model training.
    transformers==4.35.0: Hugging Face library for MobileBERT.
    coremltools==8.2: Conversion to CoreML.
    numpy==1.26.4: Data handling.)

### Model Generation
1. Run fine_tune_mobile_bert.py to generate fine_tuned_mobilebert:
    **python fine_tune_mobile_bert.py**

2. Convert to CoreML:
    **python convert_mobilebert.py**

3. Ensure that "FineTunedMobileBERT.mlpackage" file is created.

### Xcode Setup
1. Open SafeChatAIApp/SafeChatAIApp.xcodeproj in Xcode.
2. Select iPhone 16 Pro simulator (iOS 18.2 or above).
3. Add FineTunedMobileBERT.mlpackage file created by running convert_mobileBERT.py into SafeChatAIApp folder. Ensure vocab.txt is included in the directory as well (included in commit). If you encounter error running the app, make sure the .mlpackage file is in Copy Bundle Resources. 

---

## Usage

1. Launch the app in Xcode simulator.

2. Enter text (e.g., "I like", "I hate", "Fuck").

3. View real-time safety classification ("This is a safe reply!, "Input blocked for safety reasons", "Input contains inappropriate language.").

---

## Challengegs and Solutions

1. CoreML Conversion Errors: Initial attempts to convert MobileBERT to CoreML failed due to dynamic PyTorch operations (e.g., untraced control flow). Wrapped the model in MobileBERTWrapper and used torch.jit.trace with a sample input ("I like") to create a static graph, ensuring compatibility with coremltools.

2. NaN Values: Model logits occasionally produced nan values, preventing accurate classification of safe and unsafe words. Added normalization in MobileBERTWrapper.forward (logits - torch.max(logits)), stabilizing softmax outputs.

3. Tokenization mismatch: Swift-based tokenization in ContentView.swift didn’t match the Python tokenizer, leading to inconsistent predictions. Built VocabularyManager to load vocab.txt in Swift, mirroring the Python tokenizer’s behavior with [CLS], [SEP], and [UNK].

4. Performance: Initial inference was slow and blocked the UI. Introduced Swift concurrency with async/await and DispatchQueue.global in classifyInput, offloading inference to a background thread.

---

## Future Improvements

Unit Tests: Add Swift tests for classifyInput()

---

## References

- Apple Foundation Models(https://arxiv.org/pdf/2407.21075)
- MobileBERT(https://huggingface.co/google/mobilebert-uncased)
- CoreML Documentation(https://developer.apple.com/documentation/coreml)

---

## License

MIT License
