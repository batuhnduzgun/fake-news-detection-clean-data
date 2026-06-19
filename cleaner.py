import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from turkish_stop_words import TURKISH_STOP_WORDS


def extract_features(text: str) -> Dict[str, Any]:
    """Extract enhanced features from text for ML processing."""
    features = {}

    # Basic statistical features
    words = text.split()
    features["text_length"] = len(text)
    features["word_count"] = len(words)
    features["avg_word_length"] = (
        sum(len(word) for word in words) / len(words) if words else 0
    )

    # Enhanced punctuation features
    features["exclamation_count"] = text.count("!")
    features["question_count"] = text.count("?")
    features["period_count"] = text.count(".")
    features["comma_count"] = text.count(",")
    features["punctuation_ratio"] = (
        len(re.findall(r"[.,!?]", text)) / len(text) if text else 0
    )

    # Advanced lexical features
    word_counts = Counter(words)
    features["unique_words_ratio"] = len(word_counts) / len(words) if words else 0

    # Word-level features
    features["max_word_length"] = max(len(word) for word in words) if words else 0
    features["min_word_length"] = min(len(word) for word in words) if words else 0
    features["word_length_variance"] = (
        sum((len(word) - features["avg_word_length"]) ** 2 for word in words)
        / len(words)
        if words
        else 0
    )

    # Sentence features
    sentences = text.split(".")
    features["avg_sentence_length"] = (
        sum(len(s.split()) for s in sentences if s.strip()) / len(sentences)
        if sentences
        else 0
    )
    features["sentence_count"] = len([s for s in sentences if s.strip()])

    return features


def clean_tweet(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None

    # Handle Turkish uppercase-lowercase
    text = text.replace("İ", "i").replace("I", "ı").lower()

    # Remove #SONDAKİKA specifically
    text = re.sub(r"#SONDAKİKA\s*", "", text, flags=re.IGNORECASE)

    # Remove URLs and mentions
    text = re.sub(r"https?://\S+|www\.\S+|@\w+", "", text)

    # Preserve hashtag content without the # symbol
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove RT prefix
    text = re.sub(r"^rt[\s]+", "", text)

    # Basic emoji removal
    text = re.sub(r"[\U00010000-\U0010FFFF]+", "", text)

    # Preserve Turkish chars and meaningful punctuation
    text = re.sub(r"[^a-zçğıöşüâîû\s.,!?]", "", text)

    # Normalize whitespace
    text = " ".join(text.split())
    text = text.strip()

    # Remove stop words
    words = text.split()
    words = [word for word in words if word not in TURKISH_STOP_WORDS]
    text = " ".join(words)

    # Length validation
    if len(words) < 3 or len(words) > 100:
        return None

    return text


def process_text_file(input_file: str, output_file: str, features_file: str) -> None:
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read and process news
    processed_count = 0
    filtered_count = 0
    duplicate_count = 0
    all_features = []
    seen_texts: set = set()

    with (
        open(input_file, "r", encoding="utf-8") as fin,
        open(output_file, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            cleaned_text = clean_tweet(line)
            if cleaned_text:
                if cleaned_text in seen_texts:
                    duplicate_count += 1
                    continue
                seen_texts.add(cleaned_text)
                fout.write(f"{cleaned_text}\n")
                features = extract_features(cleaned_text)
                all_features.append(features)
                processed_count += 1
            else:
                filtered_count += 1

    # Save features to a separate file
    import json

    with open(features_file, "w", encoding="utf-8") as f:
        json.dump(all_features, f, ensure_ascii=False, indent=2)

    print(f"Processing complete for {output_file}")
    print(f"Processed news: {processed_count}")
    print(f"Filtered news: {filtered_count}")
    print(f"Duplicate news removed: {duplicate_count}")
    print(f"Features saved to: {features_file}")


if __name__ == "__main__":
    # Process fake news with features (all_fake_combined.txt includes external sources)
    process_text_file(
        "raw-data/fake/all_fake_combined.txt", "data/fake-news.txt", "data/fake-news-features.json"
    )

    print()

    # Process real news with features
    process_text_file(
        "raw-data/real/news.txt", "data/real-news.txt", "data/real-news-features.json"
    )
