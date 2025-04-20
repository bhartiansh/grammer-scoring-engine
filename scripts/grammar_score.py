import language_tool_python

tool = language_tool_python.LanguageTool("en-US")

def get_grammar_score(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0, 5.0  # Treat empty text as low score, high penalty
    matches = tool.check(text)
    error_count = len(matches)
    grammar_score = max(0, 5 - error_count * 0.1)  # Each error deducts 0.1
    return round(grammar_score, 2), round(error_count * 0.1, 2)
