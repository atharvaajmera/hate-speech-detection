import re
import string

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
RT_TAG_RE = re.compile(r"\bRT\b", flags=re.IGNORECASE)
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_RE = re.compile(r"\s+")


def clean_tweet(tweet: str) -> str:
    text = str(tweet)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = RT_TAG_RE.sub(" ", text)
    text = text.replace("#", "")
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip().lower()
    return text

def clean_tweet_column(df, source_col: str = "tweet", target_col: str = "clean_tweet"):
    df[target_col] = df[source_col].apply(clean_tweet)
    return df
