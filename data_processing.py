import pandas as pd


def load_data(real_news_path, fake_news_path):
    def read_and_clean(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = [
                line.strip() for line in file.readlines() if len(line.strip()) > 0
            ]  # remove empty lines

        return pd.DataFrame(lines, columns=["text"]).drop_duplicates(subset=["text"])

    # Read and clean data
    real_df = read_and_clean(real_news_path)
    fake_df = read_and_clean(fake_news_path)

    # Add labels
    real_df["is_real"] = 1
    fake_df["is_real"] = 0

    # Combine and shuffle
    df = pd.concat([real_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def prepare_balanced_data(real_news_path, fake_news_path):
    # Load data
    df = load_data(real_news_path, fake_news_path)

    # Balance the dataset
    balanced_news_count = df["is_real"].value_counts().min()
    balanced_df = pd.concat(
        [
            df[df["is_real"] == 0].head(balanced_news_count),
            df[df["is_real"] == 1].head(balanced_news_count),
        ],
        ignore_index=True,
    )

    # Split into train and test sets
    train_df = balanced_df.sample(frac=0.8, random_state=42)
    test_df = balanced_df.drop(train_df.index)

    return train_df, test_df


def prepare_imbalanced_data(real_news_path, fake_news_path, real_to_fake_ratio=0.5):
    """
    Prepare imbalanced dataset with a specified ratio of real to fake news.
    Args:
        real_news_path: Path to real news file
        fake_news_path: Path to fake news file
        real_to_fake_ratio: Ratio of real news to fake news (e.g., 0.5 means half as many real as fake)
    Returns:
        train_df, test_df: Training and test dataframes with the specified imbalance
    """
    # Load data
    df = load_data(real_news_path, fake_news_path)
    # Find minimum count to establish baseline
    min_count = df["is_real"].value_counts().min()
    # Calculate counts based on ratio
    if real_to_fake_ratio <= 1:
        fake_count = min_count
        real_count = int(min_count * real_to_fake_ratio)
    else:
        real_count = min_count
        fake_count = int(min_count / real_to_fake_ratio)
    # Get samples
    real_news = df[df["is_real"] == 1].head(real_count)
    fake_news = df[df["is_real"] == 0].head(fake_count)
    # Split into train and test sets
    train_real = real_news.sample(frac=0.8, random_state=42)
    test_real = real_news.drop(train_real.index)
    train_fake = fake_news.sample(frac=0.8, random_state=42)
    test_fake = fake_news.drop(train_fake.index)
    # Combine and shuffle
    train_df = pd.concat([train_real, train_fake], ignore_index=True)
    test_df = pd.concat([test_real, test_fake], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return train_df, test_df


def prepare_constant_fake_data(real_news_path, fake_news_path, real_multiplier=2):
    """
    Keep ALL fake news samples and place `real_multiplier` times as many real news.
    Default: all fake + 2x fake count of real (Sabit Sahte Dağılım Senaryosu).
    """
    df = load_data(real_news_path, fake_news_path)

    fake_df = df[df["is_real"] == 0]
    real_df = df[df["is_real"] == 1]

    fake_count = len(fake_df)
    real_count = min(int(fake_count * real_multiplier), len(real_df))

    combined = pd.concat(
        [fake_df, real_df.head(real_count)], ignore_index=True
    )
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = combined.sample(frac=0.8, random_state=42)
    test_df = combined.drop(train_df.index)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def prepare_imbalanced_fake_news_data(real_news_path, fake_news_path):
    """More fake news than real news (2:1 ratio)"""
    return prepare_imbalanced_data(
        real_news_path, fake_news_path, real_to_fake_ratio=0.5
    )


def prepare_imbalanced_real_news_data(real_news_path, fake_news_path):
    """More real news than fake news (2:1 ratio)"""
    return prepare_imbalanced_data(
        real_news_path, fake_news_path, real_to_fake_ratio=2.0
    )
