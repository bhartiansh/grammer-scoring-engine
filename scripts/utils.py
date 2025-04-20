def score_transcript(text):
    if not text.strip():
        return 0.0

    # Example logic (you can replace this with your actual model later)
    word_count = len(text.split())
    score = min(5.0, max(0.0, word_count / 20))  # Normalize to 0–5
    return round(score, 2)
