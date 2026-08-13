from backend.text_intelligence.pipeline import TextIntelligencePipeline

pipeline = TextIntelligencePipeline()

text = "This is awesome! How are you?"

result = pipeline.run(text)

print(result)
