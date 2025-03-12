import SwiftUI
import CoreML
import Foundation

class VocabularyManager: ObservableObject {
    @Published var vocab: [String: Int] = [:]
    
    init() {
        loadVocabulary()
    }
    
    private func loadVocabulary() {
        guard let path = Bundle.main.path(forResource: "vocab", ofType: "txt") else {
            fatalError("Vocabulary file 'vocab.txt' not found")
        }
        do {
            let vocabString = try String(contentsOfFile: path, encoding: .utf8)
            var tempVocab: [String: Int] = [:]
            let lines = vocabString.components(separatedBy: CharacterSet.newlines).filter { !$0.isEmpty }
            for (index, line) in lines.enumerated() {
                let token = line.trimmingCharacters(in: .whitespacesAndNewlines)
                if !token.isEmpty {
                    tempVocab[token] = index
                }
            }
            vocab = tempVocab
        } catch {
            fatalError("Vocabulary loading failed!: \(error.localizedDescription)")
        }
    }
}

struct ContentView: View {
    @State private var inputText = ""
    @State private var responseText = ""
    @State private var isProcessing = false
    @StateObject private var vocabManager = VocabularyManager()
    
    let model: MLModel
    
    init() {
        do {
            guard let modelURL = Bundle.main.url(forResource: "FineTunedMobileBERT", withExtension: "mlmodelc") else {
                fatalError("Could not find FineTunedMobileBERT.mlmodelc in the bundle!")
            }
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly
            let loadedModel = try MLModel(contentsOf: modelURL, configuration: config)
            self.model = loadedModel
        } catch {
            fatalError("Model loading failed!: \(error.localizedDescription)")
        }
    }
    
    var body: some View {
        VStack(spacing: 20) {
            Text("SafeChat AI")
                .font(.title)
                .fontWeight(.bold)
            
            Text(responseText)
                .font(.body)
                .foregroundColor(.gray)
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(.systemGray6))
                .cornerRadius(10)
            
            Spacer()
            
            HStack {
                TextField("Type a message here!", text: $inputText)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .disabled(isProcessing)
                
                Button(action: {
                    Task {
                        await processInput(inputText)
                    }
                }) {
                    Text("Send")
                        .fontWeight(.semibold)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(isProcessing ? Color.gray : Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
                .disabled(isProcessing || inputText.isEmpty)
            }
            .padding(.horizontal)
        }
        .padding(.vertical)
    }
    
    // MARK: - Guardrail; profanity
    private func applyGuardrails(_ input: String) -> (isSafe: Bool, message: String?) {
        let profanityPatterns = ["\\b(fuck|shit|damn)\\b"]// add more words to be filtered
        do {
            for pattern in profanityPatterns {
                let regex = try NSRegularExpression(pattern: pattern, options: .caseInsensitive)
                if regex.firstMatch(in: input, options: [], range: NSRange(input.startIndex..., in: input)) != nil {
                    return (false, "Input contains inappropriate language.")
                }
            }
            return (true, nil)
        } catch {
            return (false, "Error processing input safety.")
        }
    }
    
    // MARK: - Processing input with multi threading
    private func processInput(_ input: String) async {
        await MainActor.run {
            isProcessing = true
            responseText = "Processing..."
        }
        
        let (guardrailSafe, guardrailMessage) = applyGuardrails(input)
        let response: String
        if !guardrailSafe {
            response = guardrailMessage ?? "Input blocked by safety guardrails."
        } else {
            response = await classifyInput(input: input)
        }
        
        await MainActor.run {
            responseText = response
            isProcessing = false
            inputText = ""
        }
    }
    
    private func classifyInput(input: String) async -> String {
        let (inputIds, attentionMask) = tokenize(input: input)
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let inputDict = [
                        "input_ids": MLFeatureValue(multiArray: inputIds),
                        "attention_mask": MLFeatureValue(multiArray: attentionMask)
                    ]
                    let featureProvider = try MLDictionaryFeatureProvider(dictionary: inputDict)
                    let prediction = try model.prediction(from: featureProvider)
                    guard let probsMultiArray = prediction.featureValue(for: "probabilities")?.multiArrayValue else {
                        continuation.resume(returning: "Error: No probabilities output")
                        return
                    }
                    let unsafeProb = Double(truncating: probsMultiArray[0])
                    let safeProb = Double(truncating: probsMultiArray[1])
                    let label = safeProb > unsafeProb ? "safe" : "unsafe"
                    let result = label == "safe" ? "This is a safe reply!" : "Input blocked for safety reasons..."
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(returning: "Error: \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func tokenize(input: String) -> (MLMultiArray, MLMultiArray) {
        let maxLength = 64
        let inputIds = try! MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
        let attentionMask = try! MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
        
        var tokens: [Int] = [vocabManager.vocab["[CLS]"] ?? 101]
        let words = input.lowercased().split(separator: " ").map { String($0.replacingOccurrences(of: "[^a-zA-Z]", with: "", options: .regularExpression)) }.filter { !$0.isEmpty }
        
        for word in words {
            if let id = vocabManager.vocab[word] {
                tokens.append(id)
            } else {
                tokens.append(vocabManager.vocab["[UNK]"] ?? 100)
            }
        }
        tokens.append(vocabManager.vocab["[SEP]"] ?? 102)
        
        let finalTokens = tokens.prefix(maxLength).map { $0 }
        let paddedTokens = finalTokens + Array(repeating: vocabManager.vocab["[PAD]"] ?? 0, count: max(0, maxLength - finalTokens.count))
        let mask = Array(repeating: 1, count: finalTokens.count) + Array(repeating: 0, count: max(0, maxLength - finalTokens.count))
        
        for i in 0..<maxLength {
            inputIds[i] = NSNumber(value: paddedTokens[i])
            attentionMask[i] = NSNumber(value: mask[i])
        }
        return (inputIds, attentionMask)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
